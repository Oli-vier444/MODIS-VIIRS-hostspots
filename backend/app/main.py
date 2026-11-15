import io
import json
import re
from functools import lru_cache
from pathlib import Path

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import csv

app = FastAPI(title="FireScope API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# backend/app/main.py -> parents[0]=app, parents[1]=backend, parents[2]=raíz del proyecto
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed"

SENSORS = ["MODIS", "VIIRS"]


def sensor_file(sensor: str, year: int) -> Path:
    return DATA_PATH / sensor / f"year={year}" / "data.parquet"


def period_dates(year: int, month: int | None) -> tuple[str, str]:
    """Rango [inicio, fin) de todo el año o de un mes."""
    if month is None:
        return f"{year}-01-01", f"{year + 1}-01-01"
    start = f"{year}-{month:02d}-01"
    end = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"
    return start, end


@app.get("/")
def home():
    return {"message": "FireScope API funcionando"}


@app.get("/fires")
def get_fires(
    year: int,
    sensor: str,
    month: int | None = Query(None, description="1-12. Si se omite, devuelve todo el año"),
    country: str | None = None,
    limit: int = 1000,
    offset: int = 0,
):
    sensor = sensor.upper()

    if sensor not in SENSORS:
        raise HTTPException(400, "Invalid sensor")
    if month is not None and not 1 <= month <= 12:
        raise HTTPException(400, "Invalid month")
    if not 2000 <= year <= 2025:
        raise HTTPException(400, "Invalid year")
    if sensor == "VIIRS" and year < 2018:
        raise HTTPException(400, "VIIRS unavailable for this year")
    if not 1 <= limit <= 10000 or offset < 0:
        raise HTTPException(400, "Invalid pagination")

    file_path = sensor_file(sensor, year)
    if not file_path.exists():
        raise HTTPException(404, f"Dataset not found: {file_path}")

    start_date, end_date = period_dates(year, month)

    where = "WHERE acq_date >= ? AND acq_date < ?"
    params = [str(file_path), start_date, end_date]
    if country:
        where += " AND country = ?"
        params.append(country)

    select = "SELECT * REPLACE (CAST(acq_date AS VARCHAR) AS acq_date)"

    with duckdb.connect() as con:
        total = con.execute(
            f"SELECT COUNT(*) FROM read_parquet(?) {where}", params
        ).fetchone()[0]
        result = con.execute(
            f"{select} FROM read_parquet(?) {where} ORDER BY acq_date, acq_time LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchdf()

    records = json.loads(result.to_json(orient="records"))

    return {
        "sensor": sensor,
        "year": year,
        "month": month,
        "total": total,
        "count": len(records),
        "limit": limit,
        "offset": offset,
        "data": records,
    }


@app.get("/calendar")
def get_calendar(
    year: int,
    month: int,
    country: str | None = None,
):
    """Detecciones por día, sumando MODIS y VIIRS (cada uno desglosado).

    Si un sensor no tiene datos ese año (p. ej. VIIRS antes de 2018),
    simplemente no se incluye y se avisa en `sensors`.
    """
    if not 1 <= month <= 12:
        raise HTTPException(400, "Invalid month")
    if not 2000 <= year <= 2025:
        raise HTTPException(400, "Invalid year")

    start_date, end_date = period_dates(year, month)

    selects, params, available = [], [], []
    for sensor in SENSORS:
        file_path = sensor_file(sensor, year)
        if not file_path.exists():
            continue
        available.append(sensor)

        sel = (
            f"SELECT CAST(acq_date AS DATE) AS d, '{sensor}' AS sensor, frp, "
            "CASE WHEN TRY_CAST(confidence AS INTEGER) >= 80 "
            "OR lower(CAST(confidence AS VARCHAR)) IN ('h', 'high') "
            "THEN 1 ELSE 0 END AS is_high "
            "FROM read_parquet(?) WHERE acq_date >= ? AND acq_date < ?"
        )
        params += [str(file_path), start_date, end_date]
        if country:
            sel += " AND country = ?"
            params.append(country)
        selects.append(sel)

    if not available:
        raise HTTPException(404, f"No datasets for year {year} in {DATA_PATH}")

    query = f"""
        SELECT CAST(d AS VARCHAR) AS day,
               SUM(CASE WHEN sensor = 'MODIS' THEN 1 ELSE 0 END) AS modis,
               SUM(CASE WHEN sensor = 'VIIRS' THEN 1 ELSE 0 END) AS viirs,
               SUM(is_high) AS high,
               COALESCE(SUM(frp), 0) AS frp
        FROM ({' UNION ALL '.join(selects)})
        GROUP BY d
        ORDER BY d
    """

    with duckdb.connect() as con:
        rows = con.execute(query, params).fetchall()

    days = [
        {
            "date": day,
            "modis": int(modis),
            "viirs": int(viirs),
            "total": int(modis) + int(viirs),
            "high": int(high),
            "frp": round(float(frp), 1),
        }
        for day, modis, viirs, high, frp in rows
    ]

    return {
        "year": year,
        "month": month,
        "country": country,
        "sensors": available,
        "total": sum(d["total"] for d in days),
        "days": days,
    }


def _conf_sql() -> str:
    """Homologa la confianza a 3 niveles: low / nominal / high.

    MODIS: numérica 0-100 (NASA: <30 baja, 30-79 nominal, >=80 alta).
    VIIRS: categórica l / n / h.
    """
    num = "TRY_CAST(confidence AS INTEGER)"
    txt = "lower(CAST(confidence AS VARCHAR))"
    return (
        f"CASE WHEN {num} IS NOT NULL THEN "
        f"CASE WHEN {num} >= 80 THEN 'high' WHEN {num} >= 30 THEN 'nominal' ELSE 'low' END "
        f"WHEN {txt} IN ('h', 'high') THEN 'high' "
        f"WHEN {txt} IN ('n', 'nominal') THEN 'nominal' "
        f"WHEN {txt} IN ('l', 'low') THEN 'low' "
        "ELSE 'unknown' END"
    )


@app.get("/analysis")
def get_analysis(
    year: int,
    month: int | None = Query(None, description="1-12. Si se omite, el periodo es todo el año"),
    country: str | None = None,
):
    """Agregados para las gráficas: serie diaria, serie mensual del año,
    confianza homologada, día/noche y países. MODIS y VIIRS por separado."""
    if month is not None and not 1 <= month <= 12:
        raise HTTPException(400, "Invalid month")
    if not 2000 <= year <= 2025:
        raise HTTPException(400, "Invalid year")

    selects, available = [], []
    for sensor in SENSORS:
        file_path = sensor_file(sensor, year)
        if not file_path.exists():
            continue
        available.append(sensor)
        path = str(file_path).replace("'", "''")  # ruta construida en el servidor, no viene del usuario
        selects.append(
            f"SELECT CAST(acq_date AS DATE) AS d, '{sensor}' AS sensor, country, frp, "
            f"daynight, {_conf_sql()} AS conf FROM read_parquet('{path}')"
        )

    if not available:
        raise HTTPException(404, f"No datasets for year {year} in {DATA_PATH}")

    start, end = period_dates(year, month)
    ystart, yend = period_dates(year, None)
    extra = " AND country = ?" if country else ""
    cp = [country] if country else []
    period = "d >= ? AND d < ?"

    with duckdb.connect() as con:
        # Vista (no tabla): cada consulta lee solo las columnas que necesita del Parquet
        con.execute(f"CREATE TEMP VIEW t AS {' UNION ALL '.join(selects)}")

        daily_rows = con.execute(
            f"SELECT CAST(d AS VARCHAR), sensor, COUNT(*), COALESCE(SUM(frp), 0) "
            f"FROM t WHERE {period}{extra} GROUP BY d, sensor ORDER BY d",
            [start, end] + cp,
        ).fetchall()

        monthly_rows = con.execute(
            f"SELECT month(d), sensor, COUNT(*) FROM t WHERE {period}{extra} GROUP BY 1, 2",
            [ystart, yend] + cp,
        ).fetchall()

        conf_rows = con.execute(
            f"SELECT sensor, conf, COUNT(*) FROM t WHERE {period}{extra} GROUP BY 1, 2",
            [start, end] + cp,
        ).fetchall()

        dn_rows = con.execute(
            f"SELECT sensor, upper(CAST(daynight AS VARCHAR)), COUNT(*) FROM t "
            f"WHERE {period}{extra} GROUP BY 1, 2",
            [start, end] + cp,
        ).fetchall()

        country_rows = []
        if not country:  # con filtro de país este desglose no tiene sentido
            country_rows = con.execute(
                f"SELECT country, sensor, COUNT(*) FROM t WHERE {period} AND country IS NOT NULL GROUP BY 1, 2",
                [start, end],
            ).fetchall()

    daily = {}
    for day, sensor, n, frp in daily_rows:
        e = daily.setdefault(
            day, {"date": day, "modis": 0, "viirs": 0, "frp_modis": 0.0, "frp_viirs": 0.0}
        )
        k = sensor.lower()
        e[k] = int(n)
        e[f"frp_{k}"] = round(float(frp), 1)

    monthly = [{"month": m, "modis": 0, "viirs": 0} for m in range(1, 13)]
    for m, sensor, n in monthly_rows:
        monthly[int(m) - 1][sensor.lower()] = int(n)

    confidence = {s: {"low": 0, "nominal": 0, "high": 0, "unknown": 0} for s in available}
    for sensor, level, n in conf_rows:
        confidence[sensor][level] = int(n)

    daynight = {s: {"D": 0, "N": 0, "other": 0} for s in available}
    for sensor, v, n in dn_rows:
        daynight[sensor][v if v in ("D", "N") else "other"] += int(n)

    by_country = {}
    for name, sensor, n in country_rows:
        e = by_country.setdefault(name, {"country": name, "modis": 0, "viirs": 0})
        e[sensor.lower()] = int(n)
    countries = sorted(by_country.values(), key=lambda c: c["modis"] + c["viirs"], reverse=True)

    return {
        "year": year,
        "month": month,
        "country": country,
        "sensors": available,
        "daily": list(daily.values()),
        "monthly": monthly,
        "confidence": confidence,
        "daynight": daynight,
        "countries": countries,
    }

# ===================== DATOS ARMONIZADOS =====================
# data/harmonized/year=YYYY/*.parquet  (MODIS y VIIRS en un solo esquema:
# latitude, longitude, acq_date, acq_time, instrument, satellite, confidence (h/n/l),
# daynight, country, frp, type, version, pixel_area)
HARMONIZED_PATH = DATA_PATH.parent / "harmonized"
INSTRUMENT = "upper(CAST(instrument AS VARCHAR))"
DAYNIGHT = "upper(CAST(daynight AS VARCHAR))"


def h_files(year: int) -> list[str]:
    d = HARMONIZED_PATH / f"year={year}"
    return sorted(str(p) for p in d.rglob("*.parquet")) if d.exists() else []


def h_source(year: int) -> str:
    """Expresión SQL que lee todos los parquet armonizados de un año."""
    files = h_files(year)
    if not files:
        raise HTTPException(404, f"No harmonized dataset for year {year} in {HARMONIZED_PATH}")
    paths = ", ".join("'" + f.replace("'", "''") + "'" for f in files)  # rutas del servidor, no del usuario
    return f"read_parquet([{paths}], union_by_name = true)"


def h_filters(
    country: str | None = None,
    instrument: str | None = Query(None, description="MODIS o VIIRS. Si se omite, ambos"),
    min_confidence: str = Query("low", description="low (todas) | nominal (nominal y alta) | high (solo alta)"),
    fire_type: str = Query("all", description="all | vegetation (type = 0)"),
    daynight: str | None = Query(None, description="D o N"),
) -> dict:
    """Filtros comunes a los endpoints armonizados -> condiciones SQL y parámetros."""
    conds, params = [], []
    if country:
        conds.append("country = ?")
        params.append(country)
    if instrument:
        instrument = instrument.upper()
        if instrument not in SENSORS:
            raise HTTPException(400, "Invalid instrument")
        conds.append(f"{INSTRUMENT} = ?")
        params.append(instrument)
    if min_confidence == "nominal":
        conds.append(f"{_conf_sql()} IN ('nominal', 'high')")
    elif min_confidence == "high":
        conds.append(f"{_conf_sql()} = 'high'")
    elif min_confidence != "low":
        raise HTTPException(400, "Invalid min_confidence")
    if fire_type == "vegetation":
        conds.append('"type" = 0')
    elif fire_type != "all":
        raise HTTPException(400, "Invalid fire_type")
    if daynight:
        daynight = daynight.upper()
        if daynight not in ("D", "N"):
            raise HTTPException(400, "Invalid daynight")
        conds.append(f"{DAYNIGHT} = ?")
        params.append(daynight)
    return {
        "sql": "".join(f" AND {c}" for c in conds),
        "params": params,
        "country": country,
        "instrument": instrument,
    }


def _check_period(year: int, month: int | None) -> None:
    if month is not None and not 1 <= month <= 12:
        raise HTTPException(400, "Invalid month")
    if not 2000 <= year <= 2025:
        raise HTTPException(400, "Invalid year")


@app.get("/harmonized/years")
def harmonized_years():
    years = sorted(
        int(p.name.split("=", 1)[1])
        for p in HARMONIZED_PATH.glob("year=*")
        if p.is_dir() and p.name.split("=", 1)[1].isdigit() and any(p.rglob("*.parquet"))
    )
    if not years:
        raise HTTPException(404, f"No harmonized datasets in {HARMONIZED_PATH}")
    return {"years": years}


def h_years() -> list[int]:
    return sorted(
        int(p.name.split("=", 1)[1])
        for p in HARMONIZED_PATH.glob("year=*")
        if p.is_dir() and p.name.split("=", 1)[1].isdigit() and any(p.rglob("*.parquet"))
    )


@lru_cache(maxsize=16)
def _series(extra: str, params: tuple) -> dict:
    """Serie diaria de TODOS los años armonizados (conteo, confianza alta, FRP y área de píxel).
    Se calcula una vez por combinación de filtros y se guarda en memoria."""
    years = h_years()
    if not years:
        raise HTTPException(404, f"No harmonized datasets in {HARMONIZED_PATH}")
    paths = ", ".join("'" + f.replace("'", "''") + "'" for y in years for f in h_files(y))
    sql = f"""
        SELECT CAST(d AS VARCHAR), COUNT(*), COUNT(*) FILTER (WHERE conf = 'high'),
               COALESCE(SUM(frp), 0), COALESCE(SUM(pixel_area), 0)
        FROM (SELECT *, CAST(acq_date AS DATE) AS d, {_conf_sql()} AS conf
              FROM read_parquet([{paths}], union_by_name = true)
              WHERE 1 = 1{extra})
        GROUP BY d ORDER BY d
    """
    with duckdb.connect() as con:
        rows = con.execute(sql, list(params)).fetchall()
    return {
        "years": years,
        "first_date": rows[0][0] if rows else None,
        "last_date": rows[-1][0] if rows else None,
        "date": [r[0] for r in rows],
        "n": [int(r[1]) for r in rows],
        "high": [int(r[2]) for r in rows],
        "frp": [round(float(r[3]), 1) for r in rows],
        "area": [round(float(r[4]), 1) for r in rows],
    }


@app.get("/harmonized/series")
def harmonized_series(
    south: float | None = Query(None, description="Zona rectangular: latitud sur"),
    north: float | None = None,
    west: float | None = None,
    east: float | None = None,
    f: dict = Depends(h_filters),
):
    """Serie diaria completa (todos los años) para matrices año × mes/semana, anomalías,
    semáforo y temporada acumulada. Filtra por país (h_filters) o por rectángulo."""
    extra, params = f["sql"], list(f["params"])
    box = [south, north, west, east]
    if any(v is not None for v in box):
        if None in box:
            raise HTTPException(400, "Zone needs south, north, west and east")
        if not -90 <= south <= north <= 90:
            raise HTTPException(400, "Invalid latitude range")
        extra += " AND latitude BETWEEN ? AND ?"
        params += [south, north]
        if west <= east:
            extra += " AND longitude BETWEEN ? AND ?"
        else:  # la zona cruza el antimeridiano
            extra += " AND (longitude >= ? OR longitude <= ?)"
        params += [west, east]
    return _series(extra, tuple(params))


@app.get("/harmonized/fires")
def harmonized_fires(
    year: int,
    month: int | None = Query(None, description="1-12. Si se omite, todo el año"),
    limit: int = 1000,
    offset: int = 0,
    f: dict = Depends(h_filters),
):
    _check_period(year, month)
    if not 1 <= limit <= 10000 or offset < 0:
        raise HTTPException(400, "Invalid pagination")

    src = h_source(year)
    start, end = period_dates(year, month)
    where = f"WHERE acq_date >= ? AND acq_date < ?{f['sql']}"
    params = [start, end] + f["params"]

    with duckdb.connect() as con:
        total = con.execute(f"SELECT COUNT(*) FROM {src} {where}", params).fetchone()[0]
        df = con.execute(
            f"SELECT * REPLACE (CAST(CAST(acq_date AS DATE) AS VARCHAR) AS acq_date) FROM {src} {where} "
            "ORDER BY acq_date, acq_time LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchdf()

    records = json.loads(df.to_json(orient="records"))
    return {
        "sensor": f["instrument"] or "MODIS + VIIRS",
        "year": year,
        "month": month,
        "total": total,
        "count": len(records),
        "limit": limit,
        "offset": offset,
        "data": records,
    }


@app.get("/harmonized/fires.csv")
def harmonized_fires_csv(
    year: int,
    month: int | None = Query(None, description="1-12. Si se omite, todo el año"),
    f: dict = Depends(h_filters),
):
    """Descarga en CSV todas las detecciones de la consulta (sin paginar), transmitidas por bloques."""
    _check_period(year, month)
    src = h_source(year)
    start, end = period_dates(year, month)
    sql = (
        f"SELECT * REPLACE (CAST(CAST(acq_date AS DATE) AS VARCHAR) AS acq_date) FROM {src} "
        f"WHERE acq_date >= ? AND acq_date < ?{f['sql']} ORDER BY acq_date, acq_time"
    )
    params = [start, end] + f["params"]

    def stream():
        con = duckdb.connect()
        try:
            cur = con.execute(sql, params)
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow([c[0] for c in cur.description])
            yield "\ufeff" + buf.getvalue()          # BOM para que Excel respete los acentos
            while True:
                rows = cur.fetchmany(20000)
                if not rows:
                    break
                buf.seek(0)
                buf.truncate(0)
                writer.writerows(rows)
                yield buf.getvalue()
        finally:
            con.close()

    parts = ["firescope", str(year)]
    if month:
        parts.append(f"{month:02d}")
    if f["country"]:
        parts.append(re.sub(r"[^A-Za-z0-9_-]", "", f["country"]))
    if f["instrument"]:
        parts.append(f["instrument"])
    filename = "_".join(parts) + ".csv"
    return StreamingResponse(
        stream(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/harmonized/calendar")
def harmonized_calendar(year: int, month: int, f: dict = Depends(h_filters)):
    """Detecciones por día con desglose por instrumento, confianza homologada,
    FRP, área de píxel (km²) y día/noche, aplicando los filtros armonizados."""
    _check_period(year, month)
    src = h_source(year)
    start, end = period_dates(year, month)

    query = f"""
        SELECT CAST(d AS VARCHAR),
               COUNT(*) FILTER (WHERE {INSTRUMENT} = 'MODIS'),
               COUNT(*) FILTER (WHERE {INSTRUMENT} = 'VIIRS'),
               COUNT(*) FILTER (WHERE conf = 'high'),
               COUNT(*) FILTER (WHERE conf = 'nominal'),
               COUNT(*) FILTER (WHERE conf = 'low'),
               COALESCE(SUM(frp), 0),
               COALESCE(SUM(pixel_area), 0),
               COUNT(*) FILTER (WHERE {DAYNIGHT} = 'D'),
               COUNT(*) FILTER (WHERE {DAYNIGHT} = 'N')
        FROM (SELECT *, CAST(acq_date AS DATE) AS d, {_conf_sql()} AS conf FROM {src}
              WHERE acq_date >= ? AND acq_date < ?{f['sql']})
        GROUP BY d ORDER BY d
    """
    with duckdb.connect() as con:
        rows = con.execute(query, [start, end] + f["params"]).fetchall()
        sensors = [r[0] for r in con.execute(f"SELECT DISTINCT {INSTRUMENT} FROM {src}").fetchall() if r[0] in SENSORS]

    days = [
        {
            "date": day, "modis": int(m), "viirs": int(v), "total": int(m) + int(v),
            "high": int(hi), "nominal": int(no), "low": int(lo),
            "frp": round(float(frp), 1), "area": round(float(area), 1),
            "day": int(dd), "night": int(nn),
        }
        for day, m, v, hi, no, lo, frp, area, dd, nn in rows
    ]
    return {
        "year": year, "month": month, "country": f["country"], "sensors": sorted(sensors),
        "total": sum(d["total"] for d in days), "days": days,
    }


@app.get("/harmonized/analysis")
def harmonized_analysis(
    year: int,
    month: int | None = Query(None, description="1-12. Si se omite, el periodo es todo el año"),
    f: dict = Depends(h_filters),
):
    """Misma estructura que /analysis, más satélites, tipo de fuente y área de píxel."""
    _check_period(year, month)
    src = h_source(year)
    start, end = period_dates(year, month)
    ystart, yend = period_dates(year, None)
    extra, fp = f["sql"], f["params"]
    period = f"d >= ? AND d < ?{extra}"

    with duckdb.connect() as con:
        con.execute(
            f"CREATE TEMP VIEW t AS SELECT CAST(acq_date AS DATE) AS d, {INSTRUMENT} AS sensor, "
            f"instrument, satellite, \"type\", country, frp, pixel_area, daynight, confidence, "
            f"{_conf_sql()} AS conf FROM {src}"
        )
        q = lambda sql, p: con.execute(sql, p).fetchall()
        available = sorted(r[0] for r in q("SELECT DISTINCT sensor FROM t", []) if r[0] in SENSORS)

        daily_rows = q(f"SELECT CAST(d AS VARCHAR), sensor, COUNT(*), COALESCE(SUM(frp), 0), COALESCE(SUM(pixel_area), 0) "
                       f"FROM t WHERE {period} GROUP BY d, sensor ORDER BY d", [start, end] + fp)
        monthly_rows = q(f"SELECT month(d), sensor, COUNT(*) FROM t WHERE {period} GROUP BY 1, 2", [ystart, yend] + fp)
        conf_rows = q(f"SELECT sensor, conf, COUNT(*) FROM t WHERE {period} GROUP BY 1, 2", [start, end] + fp)
        dn_rows = q(f"SELECT sensor, upper(CAST(daynight AS VARCHAR)), COUNT(*) FROM t WHERE {period} GROUP BY 1, 2", [start, end] + fp)
        sat_rows = q(f"SELECT sensor, satellite, COUNT(*) FROM t WHERE {period} GROUP BY 1, 2 ORDER BY 3 DESC", [start, end] + fp)
        type_rows = q(f'SELECT sensor, "type", COUNT(*) FROM t WHERE {period} GROUP BY 1, 2 ORDER BY 2', [start, end] + fp)
        country_rows = [] if f["country"] else q(
            f"SELECT country, sensor, COUNT(*) FROM t WHERE {period} AND country IS NOT NULL GROUP BY 1, 2", [start, end] + fp)

    daily, area = {}, {s: 0.0 for s in available}
    for day, sensor, n, frp, a in daily_rows:
        if sensor not in SENSORS:
            continue
        e = daily.setdefault(day, {"date": day, "modis": 0, "viirs": 0, "frp_modis": 0.0, "frp_viirs": 0.0})
        k = sensor.lower()
        e[k] = int(n)
        e[f"frp_{k}"] = round(float(frp), 1)
        area[sensor] = area.get(sensor, 0.0) + float(a)

    monthly = [{"month": m, "modis": 0, "viirs": 0} for m in range(1, 13)]
    for m, sensor, n in monthly_rows:
        if sensor in SENSORS:
            monthly[int(m) - 1][sensor.lower()] = int(n)

    confidence = {s: {"low": 0, "nominal": 0, "high": 0, "unknown": 0} for s in available}
    for sensor, level, n in conf_rows:
        if sensor in confidence:
            confidence[sensor][level] = int(n)

    daynight = {s: {"D": 0, "N": 0, "other": 0} for s in available}
    for sensor, v, n in dn_rows:
        if sensor in daynight:
            daynight[sensor][v if v in ("D", "N") else "other"] += int(n)

    by_country = {}
    for name, sensor, n in country_rows:
        if sensor in SENSORS:
            by_country.setdefault(name, {"country": name, "modis": 0, "viirs": 0})[sensor.lower()] = int(n)
    countries = sorted(by_country.values(), key=lambda c: c["modis"] + c["viirs"], reverse=True)

    return {
        "year": year, "month": month, "country": f["country"], "sensors": available,
        "daily": list(daily.values()), "monthly": monthly, "confidence": confidence,
        "daynight": daynight, "countries": countries,
        "satellites": [{"instrument": s, "satellite": sat, "count": int(n)} for s, sat, n in sat_rows if s in SENSORS],
        "types": [{"instrument": s, "type": int(t), "count": int(n)} for s, t, n in type_rows if s in SENSORS and t is not None],
        "area_km2": {s: round(v, 1) for s, v in area.items()},
    }


COUNTRIES_FILE = Path(__file__).resolve().parents[2] / "data" / "countries.csv"

@app.get("/countries")
def get_countries():
    if not COUNTRIES_FILE.exists():
        raise HTTPException(404, f"Countries file not found: {COUNTRIES_FILE}")

    with open(COUNTRIES_FILE, newline="", encoding="utf-8-sig") as f:
        rows = csv.reader(f)
        next(rows, None)  # salta el encabezado
        countries = [row[-1].strip() for row in rows if row and row[-1].strip()]

    return {
        "count": len(countries),
        "countries": countries,
    }