import json
from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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

# ------------------------------------------------------------------ serie armonizada (sale de calibrar.py)
HARM_PATH = Path(__file__).resolve().parents[2] / "data" / "harmonized"


def _harmonized_files(start: int, end: int) -> list[str]:
    files = []
    for y in range(start, end + 1):
        p = HARM_PATH / f"year={y}" / "data.parquet"
        if p.exists():
            files.append(str(p).replace("'", "''"))  # rutas del servidor, no del usuario
    return files


@app.get("/harmonized/daily")
def harmonized_daily(
    start: int = Query(2000, description="primer año"),
    end: int = Query(2025, description="último año"),
    country: str | None = None,
    month: int | None = Query(None, description="1-12 para un solo mes de cada año"),
):
    """Serie diaria armonizada: MODIS antes de VIIRS, VIIRS calibrado después.

    Devuelve fecha, n_arm (celdas con fuego en escala MODIS) y frp_arm, lo mismo que piden
    las funciones de analisis_incendiosV3. Los días sin fuego no vienen: cuentan como cero.
    """
    if not 2000 <= start <= end <= 2025:
        raise HTTPException(400, "Invalid year range")
    if month is not None and not 1 <= month <= 12:
        raise HTTPException(400, "Invalid month")
    files = _harmonized_files(start, end)
    if not files:
        raise HTTPException(404, f"No harmonized data in {HARM_PATH}. Run calibrar.py first.")

    where, params = [], []
    if country:
        where.append("pais = ?")
        params.append(country)
    if month:
        where.append("month(fecha) = ?")
        params.append(month)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    lista = ", ".join(f"'{f}'" for f in files)

    query = f"""
        SELECT CAST(CAST(fecha AS DATE) AS VARCHAR) AS fecha,
               SUM(n_arm)   AS n_arm,
               SUM(frp_arm) AS frp_arm,
               SUM(n_modis) AS n_modis,
               SUM(n_viirs) AS n_viirs,
               MAX(fuente)  AS fuente
        FROM read_parquet([{lista}])
        {where_sql}
        GROUP BY 1
        ORDER BY 1
    """
    with duckdb.connect() as con:
        rows = con.execute(query, params).fetchall()

    days = [
        {"fecha": f, "n_arm": round(float(n), 2), "frp_arm": round(float(p), 1),
         "n_modis": int(nm), "n_viirs": int(nv), "fuente": fu}
        for f, n, p, nm, nv, fu in rows
    ]
    return {"start": start, "end": end, "country": country, "month": month,
            "total_n_arm": round(sum(d["n_arm"] for d in days), 1), "days": days}


@app.get("/harmonized/calibration")
def harmonized_calibration():
    """Parámetros y métricas de la calibración, para mostrarlos en la app."""
    meta = Path(__file__).resolve().parents[2] / "data" / "calibracion" / "meta.json"
    if not meta.exists():
        raise HTTPException(404, "Run calibrar.py first.")
    return json.loads(meta.read_text(encoding="utf-8"))
