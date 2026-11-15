"""
Calibración y armonización MODIS / VIIRS sobre los parquet de data/processed.

Lee:    data/processed/{MODIS,VIIRS}/year=AAAA/data.parquet   (puntos crudos, como los deja el notebook)
Escribe:
    data/celdas/{MODIS,VIIRS}/year=AAAA.parquet    celdas de 0.01° por país y día (pesan mucho menos)
    data/calibracion/factores.csv                  factor MODIS/VIIRS por zona de 5° y mes (conteo y FRP)
    data/calibracion/factor_global.csv             factor global por mes (respaldo para zonas sin datos)
    data/calibracion/validacion_mensual.csv        MODIS, VIIRS crudo y VIIRS calibrado por mes
    data/calibracion/validacion.png                la gráfica para la presentación
    data/calibracion/meta.json                     parámetros usados y métricas
    data/harmonized/year=AAAA/data.parquet         serie armonizada: pais, fecha, ci, cj (0.1°),
                                                   n_arm, frp_arm, n_modis, n_viirs, fuente

Uso (desde la raíz del proyecto):
    python calibrar.py                      # todo
    python calibrar.py --etapa celdas       # solo pasar puntos a celdas (lo más lento, se hace una vez)
    python calibrar.py --etapa calibrar     # factores y validación
    python calibrar.py --etapa armonizar    # serie final que lee la API

Idea: cada punto pasa a una celda de 0.01°. Con los años en que ambos sensores existen se calcula,
por zona de 5° y mes, cuántas celdas ve MODIS por cada celda que ve VIIRS. Desde que empieza VIIRS
se usa VIIRS por ese factor; antes, MODIS tal cual. Así la serie no salta cuando entra VIIRS.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
    HAY_PYARROW = True
except ImportError:                       # solo para pruebas sin pyarrow
    HAY_PYARROW = False

SENSORES = ("MODIS", "VIIRS")
COLUMNAS = ["latitude", "longitude", "acq_date", "confidence", "type", "frp", "country"]
CONF_LETRAS = {"n", "h", "nominal", "high"}


# ------------------------------------------------------------------ E/S
def archivo_sensor(base, sensor, anio):
    for nombre in ("data.parquet", "data.csv"):
        p = Path(base) / sensor / f"year={anio}" / nombre
        if p.exists():
            return p
    return None


def anios_disponibles(base, sensor):
    d = Path(base) / sensor
    if not d.exists():
        return []
    return sorted(int(x.name.split("=")[1]) for x in d.glob("year=*") if archivo_sensor(base, sensor, x.name.split("=")[1]))


def lotes(ruta, tam=1_000_000):
    """Lee el archivo por partes y solo con las columnas necesarias, para no llenar la memoria."""
    ruta = Path(ruta)
    if ruta.suffix == ".parquet":
        if not HAY_PYARROW:
            raise RuntimeError("Instala pyarrow para leer parquet: pip install pyarrow")
        f = pq.ParquetFile(ruta)
        cols = [c for c in COLUMNAS if c in f.schema_arrow.names]
        for b in f.iter_batches(batch_size=tam, columns=cols):
            yield b.to_pandas()
    else:
        yield from pd.read_csv(ruta, chunksize=tam, dtype={"confidence": str},
                               usecols=lambda c: c in COLUMNAS)


def anios_celdas(data, sensor):
    d = Path(data) / "celdas" / sensor
    if not d.exists():
        return []
    return sorted({int(x.stem.split("=")[1]) for x in d.glob("year=*") if x.suffix in (".parquet", ".csv")})


def guardar(df, ruta):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if HAY_PYARROW:
        df.to_parquet(ruta.with_suffix(".parquet"), index=False, compression="zstd")
    else:
        df.to_csv(ruta.with_suffix(".csv"), index=False)


def leer(ruta_sin_ext):
    p = Path(ruta_sin_ext)
    if p.with_suffix(".parquet").exists():
        return pd.read_parquet(p.with_suffix(".parquet"))
    if p.with_suffix(".csv").exists():
        return pd.read_csv(p.with_suffix(".csv"), parse_dates=["fecha"])
    return None


# ------------------------------------------------------------------ etapa 1: puntos -> celdas
def indice_celda(v, escala=100):
    """floor(v*escala) sin error de punto flotante (40.12*100 = 4011.999...)."""
    return np.floor(np.round(np.asarray(v, dtype=float) * escala, 6)).astype("int32")


def limpiar(df, conf_min=30):
    """Misma regla para ambos sensores: confianza >= 30 o n/h, type 0 (o vacío), coordenadas y FRP válidos."""
    conf = df["confidence"]
    if pd.api.types.is_numeric_dtype(conf):
        ok = conf >= conf_min
    else:
        c = conf.astype(str).str.strip().str.lower()
        ok = (pd.to_numeric(c, errors="coerce") >= conf_min) | c.isin(CONF_LETRAS)
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    frp = pd.to_numeric(df["frp"], errors="coerce")
    fecha = pd.to_datetime(df["acq_date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    ok &= lat.between(-90, 90) & lon.between(-180, 180) & frp.notna() & fecha.notna()
    if "type" in df.columns:
        t = pd.to_numeric(df["type"], errors="coerce")
        ok &= t.isna() | t.eq(0)
    m = ok.to_numpy()
    pais = (df["country"].astype("string").fillna("Desconocido") if "country" in df.columns
            else pd.Series("Desconocido", index=df.index))
    return pd.DataFrame({
        "pais": pais.to_numpy()[m],
        "fecha": fecha.to_numpy()[m],
        "ci": indice_celda(lat.to_numpy()[m]),
        "cj": indice_celda(lon.to_numpy()[m]),
        "frp": np.clip(frp.to_numpy()[m], 0, None),
    })


def a_celdas(df):
    return df.groupby(["pais", "fecha", "ci", "cj"], as_index=False).agg(frp=("frp", "sum"),
                                                                        puntos=("frp", "size"))


def etapa_celdas(args):
    for sensor in SENSORES:
        for anio in anios_disponibles(args.processed, sensor):
            destino = Path(args.data) / "celdas" / sensor / f"year={anio}"
            if not args.rehacer and (destino.with_suffix(".parquet").exists() or destino.with_suffix(".csv").exists()):
                print(f"{sensor} {anio}: ya existe, se salta (usa --rehacer para repetir)")
                continue
            partes, leidas, quedan = [], 0, 0
            for lote in lotes(archivo_sensor(args.processed, sensor, anio)):
                leidas += len(lote)
                limpio = limpiar(lote, args.conf_min)
                quedan += len(limpio)
                partes.append(a_celdas(limpio))
            # una celda-día puede repartirse entre lotes: se vuelve a juntar
            celdas = (pd.concat(partes, ignore_index=True)
                      .groupby(["pais", "fecha", "ci", "cj"], as_index=False)
                      .agg(frp=("frp", "sum"), puntos=("puntos", "sum")))
            celdas["frp"] = celdas["frp"].astype("float32")
            celdas["puntos"] = celdas["puntos"].astype("int32")
            guardar(celdas, destino)
            print(f"{sensor} {anio}: {leidas:,} puntos, {quedan:,} tras filtro, {len(celdas):,} celdas-día")


# ------------------------------------------------------------------ etapa 2: calibración
def celdas_sensor(args, sensor, anio):
    return leer(Path(args.data) / "celdas" / sensor / f"year={anio}")


def conteos_zona(celdas):
    """Celdas-día únicas (una celda de frontera cuenta una vez) por zona de 5° y mes."""
    u = celdas.groupby(["fecha", "ci", "cj"], as_index=False)["frp"].sum()
    u["zi"], u["zj"], u["mes"] = u["ci"] // 500, u["cj"] // 500, u["fecha"].dt.month
    return u.groupby(["zi", "zj", "mes"], as_index=False).agg(n=("ci", "size"), frp=("frp", "sum"))


def calcular_factores(cnt, anios, a_n):
    c = cnt[cnt["anio"].isin(anios)]
    piv = c.pivot_table(index=["zi", "zj", "mes"], columns="sensor", values=["n", "frp"],
                        aggfunc="sum", fill_value=0)
    piv.columns = [f"{v}_{s}" for v, s in piv.columns]
    piv = piv.reset_index()
    for col in ("n_MODIS", "n_VIIRS", "frp_MODIS", "frp_VIIRS"):
        if col not in piv:
            piv[col] = 0.0

    g = piv.groupby("mes")[["n_MODIS", "n_VIIRS", "frp_MODIS", "frp_VIIRS"]].sum()
    if (g["n_VIIRS"] == 0).any() or (g["n_MODIS"] == 0).any():
        faltan = g.index[(g["n_VIIRS"] == 0) | (g["n_MODIS"] == 0)].tolist()
        raise SystemExit(f"Los años de calibración {anios} no tienen datos de ambos sensores en los meses {faltan}.")
    fg = pd.DataFrame({"mes": g.index, "fg_n": g["n_MODIS"] / g["n_VIIRS"],
                       "fg_frp": g["frp_MODIS"] / g["frp_VIIRS"]}).reset_index(drop=True)

    # suavizado hacia el factor global del mes; el peso del FRP equivale al de conteos
    frp_medio = g["frp_VIIRS"].sum() / g["n_VIIRS"].sum()
    a_frp = a_n * frp_medio
    piv = piv.merge(fg, on="mes")
    piv["factor_n"] = (piv["n_MODIS"] + a_n * piv["fg_n"]) / (piv["n_VIIRS"] + a_n)
    piv["factor_frp"] = (piv["frp_MODIS"] + a_frp * piv["fg_frp"]) / (piv["frp_VIIRS"] + a_frp)
    return piv[["zi", "zj", "mes", "n_MODIS", "n_VIIRS", "factor_n", "factor_frp"]], fg


def factor_para(df, fac, fg, col):
    """Factor de cada fila según zona y mes; si la zona no se vio en la calibración, el global del mes."""
    f = df[["zi", "zj", "mes"]].merge(fac[["zi", "zj", "mes", col]], how="left", on=["zi", "zj", "mes"])[col]
    glob = df["mes"].map(fg.set_index("mes")["fg_" + col.split("_")[1]])
    return f.fillna(glob.reset_index(drop=True)).to_numpy()


def etapa_calibrar(args):
    filas, inicio_viirs = [], None
    anios_v = anios_celdas(args.data, "VIIRS")
    anios_m = anios_celdas(args.data, "MODIS")
    traslape = sorted(set(anios_v) & set(anios_m))
    if not traslape:
        raise SystemExit("No hay años con MODIS y VIIRS a la vez en data/celdas. Corre antes --etapa celdas.")
    for anio in traslape:
        for sensor in SENSORES:
            cel = celdas_sensor(args, sensor, anio)
            if sensor == "VIIRS":
                d0 = cel["fecha"].min()
                inicio_viirs = d0 if inicio_viirs is None else min(inicio_viirs, d0)
            c = conteos_zona(cel)
            c["anio"], c["sensor"] = anio, sensor
            filas.append(c)
    cnt = pd.concat(filas, ignore_index=True)

    anios_cal = [a for a in traslape if args.cal_ini <= a <= args.cal_fin]
    if not anios_cal:
        raise SystemExit(f"Ningún año de traslape {traslape} cae entre {args.cal_ini} y {args.cal_fin}.")
    fac, fg = calcular_factores(cnt, anios_cal, args.suavizado)

    out = Path(args.data) / "calibracion"
    out.mkdir(parents=True, exist_ok=True)
    fac.to_csv(out / "factores.csv", index=False)
    fg.to_csv(out / "factor_global.csv", index=False)

    # validación: años de calibración y años de prueba (traslape fuera del rango)
    v = cnt.pivot_table(index=["anio", "zi", "zj", "mes"], columns="sensor", values=["n", "frp"],
                        aggfunc="sum", fill_value=0)
    v.columns = [f"{a}_{s}" for a, s in v.columns]
    v = v.reset_index()
    v["n_VIIRS_cal"] = v["n_VIIRS"] * factor_para(v, fac, fg, "factor_n")
    v["frp_VIIRS_cal"] = v["frp_VIIRS"] * factor_para(v, fac, fg, "factor_frp")
    v["muestra"] = np.where(v["anio"].isin(anios_cal), "calibracion", "prueba")
    mens = v.groupby(["muestra", "anio", "mes"], as_index=False)[
        ["n_MODIS", "n_VIIRS", "n_VIIRS_cal", "frp_MODIS", "frp_VIIRS", "frp_VIIRS_cal"]].sum()
    # el primer año de VIIRS puede empezar a medio año: se quitan los meses sin VIIRS
    mens = mens[mens["n_VIIRS"] > 0]
    mens.to_csv(out / "validacion_mensual.csv", index=False)

    metricas = {}
    for m, g in mens.groupby("muestra"):
        metricas[m] = {}
        for var in ("n", "frp"):
            mo, vr, vc = g[f"{var}_MODIS"], g[f"{var}_VIIRS"], g[f"{var}_VIIRS_cal"]
            metricas[m][var] = {
                "meses": int(len(g)),
                "r_calibrado": round(float(np.corrcoef(mo, vc)[0, 1]), 3) if len(g) > 2 else None,
                "sesgo_crudo_pct": round(float(100 * (vr.sum() / mo.sum() - 1)), 1),
                "sesgo_calibrado_pct": round(float(100 * (vc.sum() / mo.sum() - 1)), 1),
            }
    meta = {"viirs_inicio": str(inicio_viirs.date()), "anios_traslape": traslape,
            "anios_calibracion": anios_cal, "suavizado": args.suavizado, "conf_min": args.conf_min,
            "factor_n_rango": [round(float(fac.factor_n.min()), 3), round(float(fac.factor_n.max()), 3)],
            "factor_frp_rango": [round(float(fac.factor_frp.min()), 3), round(float(fac.factor_frp.max()), 3)],
            "metricas": metricas}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"VIIRS empieza: {meta['viirs_inicio']} | traslape: {traslape} | calibración: {anios_cal}")
    print(f"Factor conteo global por mes: {fg['fg_n'].round(2).tolist()}")
    print(f"Rango factor zona-mes: conteo {meta['factor_n_rango']}, FRP {meta['factor_frp_rango']}")
    for m, d in metricas.items():
        for var, x in d.items():
            print(f"  {m:12s} {var:4s} r={x['r_calibrado']} sesgo crudo {x['sesgo_crudo_pct']:+.1f}% "
                  f"-> calibrado {x['sesgo_calibrado_pct']:+.1f}%")
    grafica_validacion(mens, out / "validacion.png")


def grafica_validacion(mens, ruta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m = mens.sort_values(["anio", "mes"])
    x = pd.to_datetime(dict(year=m["anio"], month=m["mes"], day=1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, m["n_VIIRS"], color="#999999", lw=1.5, label="VIIRS sin calibrar")
    ax.plot(x, m["n_MODIS"], color="#0072B2", lw=2.2, label="MODIS")
    ax.plot(x, m["n_VIIRS_cal"], color="#D55E00", lw=2, ls="--", label="VIIRS calibrado")
    # los años de prueba pueden no ser seguidos (2018 y 2022 en adelante): se sombrea cada uno
    for i, anio in enumerate(sorted(m.loc[m["muestra"] == "prueba", "anio"].unique())):
        xa = x[(m["anio"] == anio).to_numpy()]
        ax.axvspan(xa.min(), xa.max() + pd.offsets.MonthEnd(0), color="#eeeeee", zorder=0,
                   label="Años de prueba" if i == 0 else None)
    ax.set_ylabel("Celdas con fuego por mes")
    ax.set_title("MODIS contra VIIRS antes y después de calibrar", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ etapa 3: serie armonizada
def etapa_armonizar(args):
    cal = Path(args.data) / "calibracion"
    meta = json.loads((cal / "meta.json").read_text())
    inicio_viirs = pd.Timestamp(meta["viirs_inicio"])
    fac, fg = pd.read_csv(cal / "factores.csv"), pd.read_csv(cal / "factor_global.csv")

    anios = sorted(set(anios_celdas(args.data, "MODIS")) |
                   set(anios_celdas(args.data, "VIIRS")))
    for anio in anios:
        partes = {}
        for sensor in SENSORES:
            cel = celdas_sensor(args, sensor, anio)
            if cel is None:
                continue
            # a 0.1°: n = celdas de 0.01° con fuego dentro de la celda gruesa
            g = (cel.assign(ci=cel["ci"] // 10, cj=cel["cj"] // 10)
                 .groupby(["pais", "fecha", "ci", "cj"], as_index=False)
                 .agg(n=("frp", "size"), frp=("frp", "sum")))
            partes[sensor] = g.rename(columns={"n": f"n_{sensor.lower()}", "frp": f"frp_{sensor.lower()}"})
        if not partes:
            continue
        llaves = ["pais", "fecha", "ci", "cj"]
        h = partes.get("MODIS", pd.DataFrame(columns=llaves))
        if "VIIRS" in partes:
            h = h.merge(partes["VIIRS"], how="outer", on=llaves) if "MODIS" in partes else partes["VIIRS"]
        for c in ("n_modis", "n_viirs", "frp_modis", "frp_viirs"):
            if c not in h:
                h[c] = 0
            h[c] = h[c].fillna(0)

        h["zi"], h["zj"], h["mes"] = h["ci"] // 50, h["cj"] // 50, h["fecha"].dt.month
        usa_viirs = (h["fecha"] >= inicio_viirs).to_numpy()
        f_n = factor_para(h, fac, fg, "factor_n")
        f_frp = factor_para(h, fac, fg, "factor_frp")
        h["n_arm"] = np.where(usa_viirs, h["n_viirs"] * f_n, h["n_modis"]).astype("float32")
        h["frp_arm"] = np.where(usa_viirs, h["frp_viirs"] * f_frp, h["frp_modis"]).astype("float32")
        h["fuente"] = np.where(usa_viirs, "VIIRS_calibrado", "MODIS")
        h = h[h["n_arm"] > 0]
        cols = ["pais", "fecha", "ci", "cj", "n_arm", "frp_arm", "n_modis", "n_viirs", "fuente"]
        h = h[cols].astype({"ci": "int16", "cj": "int16", "n_modis": "int32", "n_viirs": "int32"})
        guardar(h.sort_values(["pais", "fecha", "ci", "cj"]), Path(args.data) / "harmonized" / f"year={anio}" / "data")
        print(f"armonizado {anio}: {len(h):,} filas ({h['fuente'].value_counts().to_dict()})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="carpeta data del proyecto")
    ap.add_argument("--processed", default=None, help="por defecto data/processed")
    ap.add_argument("--etapa", default="todo", choices=["todo", "celdas", "calibrar", "armonizar"])
    ap.add_argument("--cal-ini", type=int, default=2019, help="primer año para calcular el factor")
    ap.add_argument("--cal-fin", type=int, default=2021, help="último año para calcular el factor")
    ap.add_argument("--suavizado", type=float, default=1000, help="peso del factor global en zonas con pocos datos")
    ap.add_argument("--conf-min", type=int, default=30)
    ap.add_argument("--rehacer", action="store_true", help="vuelve a generar celdas aunque ya existan")
    args = ap.parse_args()
    args.processed = args.processed or str(Path(args.data) / "processed")
    if args.etapa in ("todo", "celdas"):
        etapa_celdas(args)
    if args.etapa in ("todo", "calibrar"):
        etapa_calibrar(args)
    if args.etapa in ("todo", "armonizar"):
        etapa_armonizar(args)


if __name__ == "__main__":
    main()
