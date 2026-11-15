"""
Análisis de patrones y gráficas sobre una serie diaria de actividad de incendios.

Flujo completo:
    parquet o CSV de puntos -> leer_puntos -> limpiar_puntos -> puntos_a_celdas -> serie_region
    -> análisis y gráficas

La entrada de los análisis es una tabla con una fila por día de una región:
    fecha, n_arm (celdas con fuego armonizadas), frp_arm (energía total armonizada)
Los días sin incendio deben contar como cero, por eso serie_region() y preparar() rellenan los huecos.

Cada análisis es una función que devuelve datos (DataFrame o diccionario) y cada gráfica
es otra función que recibe esos datos y devuelve una figura. La API puede llamar solo
a las primeras, y la interfaz o el CSV de descarga usan sus resultados.

Pruebas rápidas:
    python analisis_incendioV2.py                                      # datos simulados
    python analisis_incendioV2.py --puntos modis.parquet --pais Peru    # datos reales
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # sin ventana, para poder guardar imágenes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

AZUL, NARANJA, GRIS = "#00F2FF", "#FB730A", "#6b6b6b"   # par de colores seguro para daltonismo
MIN_ANIOS_REF = 5                                          # historia mínima para comparar


# ---------------------------------------------------------------- datos de prueba
def generar_serie_falsa(inicio="2000-11-01", fin="2026-11-14", semilla=1):
    """Serie diaria inventada: temporada alta en septiembre, años fuertes y poca tendencia."""
    rng = np.random.default_rng(semilla)
    dias = pd.date_range(inicio, fin, freq="D")
    doy = dias.dayofyear.to_numpy()
    temporada = 1 + 0.9 * np.sin((doy - 170) / 365 * 2 * np.pi)       # pico hacia septiembre
    efecto_anio = {a: rng.lognormal(0, 0.35) for a in range(dias.year.min(), dias.year.max() + 1)}
    for a in (2005, 2010, 2019, 2024):                                  # años excepcionales
        if a in efecto_anio:
            efecto_anio[a] *= 1.8
    f_anio = dias.year.map(efecto_anio).to_numpy()
    tendencia = 1 + 0.008 * (dias.year.to_numpy() - 2000)
    n = rng.poisson(np.clip(40 * temporada**2 * f_anio * tendencia, 0.2, None))
    frp = n * rng.gamma(4, 6, len(dias))
    return pd.DataFrame({"fecha": dias, "n_arm": n.astype(float), "frp_arm": frp})


# ---------------------------------------------------------------- de puntos a serie diaria
COLUMNAS_PUNTOS = ["latitude", "longitude", "acq_date", "confidence", "type", "frp", "country"]
CONFIANZA_LETRAS = {"n", "h", "nominal", "high"}      # VIIRS: nominal y alta


def leer_puntos(ruta, columnas=COLUMNAS_PUNTOS):
    """
    Lee solo las columnas que se usan. En el parquet de ejemplo, latitud y longitud en float64
    ocupan casi dos tercios del archivo, así que conviene no cargar nada de más.
    """
    ruta = str(ruta)
    if ruta.endswith(".parquet"):
        try:
            return pd.read_parquet(ruta, columns=list(columnas))
        except Exception:                               # por ejemplo, si el archivo no trae country
            return pd.read_parquet(ruta, columns=[c for c in columnas if c != "country"])
    return pd.read_csv(ruta, usecols=lambda c: c in columnas, dtype={"confidence": str})


def limpiar_puntos(df, conf_min=30):
    """
    Aplica la misma regla de calidad para los dos sensores:
    confianza de 30 o más (MODIS) o nominal y alta (VIIRS), solo incendios de vegetación (type 0).
    """
    faltantes = {"latitude", "longitude", "acq_date", "confidence", "frp"}.difference(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(faltantes))}")

    if pd.api.types.is_numeric_dtype(df["confidence"]):
        ok = df["confidence"] >= conf_min
    else:
        c = df["confidence"].astype(str).str.strip().str.lower()
        ok = (pd.to_numeric(c, errors="coerce") >= conf_min) | c.isin(CONFIANZA_LETRAS)

    latitud = pd.to_numeric(df["latitude"], errors="coerce")
    longitud = pd.to_numeric(df["longitude"], errors="coerce")
    frp = pd.to_numeric(df["frp"], errors="coerce")
    fechas = pd.to_datetime(df["acq_date"], errors="coerce", utc=True)
    ok &= latitud.between(-90, 90).fillna(False)
    ok &= longitud.between(-180, 180).fillna(False)
    ok &= np.isfinite(frp.to_numpy(dtype=float))
    ok &= fechas.notna()
    if "type" in df.columns:
        ok &= pd.to_numeric(df["type"], errors="coerce").eq(0).fillna(False)

    d = df.loc[ok].copy()
    d["latitude"] = latitud.loc[ok]
    d["longitude"] = longitud.loc[ok]
    d["acq_date"] = fechas.loc[ok]
    d["frp"] = frp.loc[ok].clip(lower=0)                 # el parquet trae algún -0.0
    return d


def puntos_a_celdas(df):
    """
    Una fila por celda de 0.01 grados y día (con país, si existe).
    La celda se identifica con índices enteros: ci = floor(lat*100), cj = floor(lon*100).
    """
    d = pd.DataFrame({
        "fecha": pd.to_datetime(df["acq_date"]).dt.normalize().to_numpy(),
        "ci": np.floor(df["latitude"].to_numpy() * 100).astype("int32"),
        "cj": np.floor(df["longitude"].to_numpy() * 100).astype("int32"),
        "frp": df["frp"].to_numpy(),
    })
    llaves = ["fecha", "ci", "cj"]
    if "country" in df.columns:
        d["pais"] = df["country"].astype(str).to_numpy()
        llaves = ["pais"] + llaves
    return d.groupby(llaves, as_index=False).agg(puntos=("frp", "size"), frp=("frp", "sum"))


def top_paises(celdas, n=10):
    """Países con más celdas-día de fuego: sirve para el selector de la app."""
    return (celdas.groupby("pais").size().sort_values(ascending=False).head(n)
            .rename("celdas_dia").reset_index())


def serie_region(celdas, pais=None, bbox=None):
    """
    Serie diaria de una región (país o caja lat/lon) con ceros en los días sin fuego.
    bbox = (lat_min, lat_max, lon_min, lon_max). Devuelve fecha, n_arm y frp_arm.
    Aquí n_arm cuenta celdas-día sin calibrar: el factor MODIS/VIIRS se aplica antes, en la calibración.
    """
    c = celdas
    if pais is not None:
        if "pais" not in c.columns:
            raise ValueError("La tabla no tiene país; usa bbox.")
        c = c[c["pais"] == pais]
    if bbox is not None:
        lat0, lat1, lon0, lon1 = bbox
        c = c[(c["ci"] >= np.floor(lat0 * 100)) & (c["ci"] < np.floor(lat1 * 100)) &
              (c["cj"] >= np.floor(lon0 * 100)) & (c["cj"] < np.floor(lon1 * 100))]
    if c.empty:
        raise ValueError("La región no tiene detecciones.")

    # El calendario se completa con el periodo que cubren los datos, no con el de la región,
    # si no una región cuyo primer fuego fue en febrero perdería enero.
    cobertura = pd.date_range(celdas["fecha"].min(), celdas["fecha"].max(), freq="D")
    s = c.groupby("fecha").agg(n_arm=("ci", "size"), frp_arm=("frp", "sum"))
    s = s.reindex(cobertura, fill_value=0.0).rename_axis("fecha").reset_index()
    s["n_arm"] = s["n_arm"].astype(float)
    return s


# ---------------------------------------------------------------- utilidades
def preparar(serie):
    """Fecha como índice diario, con ceros en los días sin registro."""
    # Comprueba el esquema antes de seleccionar columnas o agrupar.
    requeridas = {"fecha", "n_arm", "frp_arm"}
    faltantes = requeridas.difference(serie.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(sorted(faltantes))}")
    if serie.empty:
        raise ValueError("La serie no contiene filas.")

    # Convierte fechas con zona horaria o sin ella a días UTC comparables.
    s = serie.loc[:, ["fecha", "n_arm", "frp_arm"]].copy()
    s["fecha"] = pd.to_datetime(s["fecha"], errors="raise", utc=True)
    s["fecha"] = s["fecha"].dt.tz_convert(None).dt.normalize()
    if s["fecha"].isna().any():
        raise ValueError("La columna fecha contiene valores vacíos o no válidos.")

    # Rechaza valores no numéricos, faltantes, infinitos o negativos.
    for col in ["n_arm", "frp_arm"]:
        s[col] = pd.to_numeric(s[col], errors="raise")
        valores = s[col].to_numpy(dtype=float)
        if not np.isfinite(valores).all() or (valores < 0).any():
            raise ValueError(f"{col} debe contener solo valores finitos y no negativos.")

    # Suma registros del mismo día y completa los días sin detecciones con cero.
    s = s.groupby("fecha")[["n_arm", "frp_arm"]].sum().sort_index()
    calendario = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(calendario, fill_value=0.0)


# Rechaza métricas desconocidas en vez de calcular o etiquetar una columna equivocada.
def _validar_col(col):
    if col not in {"n_arm", "frp_arm"}:
        raise ValueError("col debe ser 'n_arm' o 'frp_arm'.")


def _meses_completos(s):
    """Total mensual solo de los meses cubiertos por completo (el primero y el último suelen no estarlo)."""
    mensual = s.resample("MS").sum()
    inicio, fin = s.index.min(), s.index.max()
    ok = (mensual.index >= inicio) & ((mensual.index + pd.offsets.MonthEnd(0)) <= fin)
    return mensual[ok]


def _anios_completos(s):
    inicio, fin = s.index.min(), s.index.max()
    return [a for a in range(inicio.year, fin.year + 1)
            if pd.Timestamp(a, 1, 1) >= inicio and pd.Timestamp(a, 12, 31) <= fin]


# ---------------------------------------------------------------- análisis (devuelven datos)
def estacionalidad(serie, col="n_arm"):
    """Total mensual típico: media y percentiles 10 y 90 entre años, por mes."""
    _validar_col(col)
    m = _meses_completos(preparar(serie))[col]
    g = m.groupby(m.index.month)
    return pd.DataFrame({"media": g.mean(), "p10": g.quantile(0.10), "p90": g.quantile(0.90),
                         "anios": g.size()}).rename_axis("mes")


def totales_anuales(serie, col="n_arm"):
    """Total por año completo y tendencia (pendiente de Theil-Sen, más robusta a años extremos)."""
    _validar_col(col)
    s = preparar(serie)
    anual = s[col].groupby(s.index.year).sum()
    anual = anual[anual.index.isin(_anios_completos(s))].rename("total")
    if len(anual) < MIN_ANIOS_REF:
        return anual.to_frame(), {"pendiente_por_anio": np.nan, "cambio_pct_por_decada": np.nan}
    pend, ord_, _, _ = theilslopes(anual.to_numpy(), anual.index.to_numpy())
    ref = anual.median()
    return anual.to_frame(), {"pendiente_por_anio": pend, "ordenada": ord_,
                              "cambio_pct_por_decada": 100 * pend * 10 / ref if ref else np.nan}


def anomalias_mensuales(serie, col="n_arm"):
    """
    Anomalía de cada mes contra el promedio de ese mismo mes en los demás años (sin incluir el propio).
    Devuelve valor, referencia, anomalía (diferencia) y z (en desviaciones estándar).
    """
    _validar_col(col)
    m = _meses_completos(preparar(serie))[col]
    x = m.groupby([m.index.year, m.index.month]).sum().unstack()          # años x meses
    cnt = x.notna().sum()
    s1, s2 = x.sum(), (x ** 2).sum()
    n_ex = cnt - 1
    suma_ex = s1 - x
    media_ex = suma_ex.div(n_ex.where(n_ex > 0))
    var_ex = (s2 - x ** 2 - suma_ex ** 2 / n_ex.where(n_ex > 0)).div(
        (n_ex - 1).where(n_ex > 1)
    )
    std_ex = np.sqrt(var_ex.clip(lower=0))
    ok = n_ex >= MIN_ANIOS_REF
    out = pd.DataFrame({
        "valor": x.stack(),
        "referencia": media_ex.where(ok, np.nan).stack(),
        "std_ref": std_ex.where(ok, np.nan).stack(),
    })
    out["anomalia"] = out["valor"] - out["referencia"]
    out["z"] = out["anomalia"] / out["std_ref"].replace(0, np.nan)
    out.index.names = ["anio", "mes"]
    return out.reset_index()


def acumulado_temporada(serie, anio, col="n_arm"):
    """
    Curva acumulada del año elegido contra los demás años completos.
    Devuelve las curvas, las bandas p10/p50/p90 y el lugar del año a la fecha de corte.
    """
    _validar_col(col)
    s = preparar(serie)
    d = s[col].to_frame("v")
    d["anio"] = d.index.year
    # Alinea años bisiestos y no bisiestos en un calendario común de 366 días.
    ajuste = (~d.index.is_leap_year) & (d.index.month > 2)
    d["doy"] = d.index.dayofyear + ajuste.astype(int)
    por_dia = d.groupby(["anio", "doy"])["v"].sum().unstack(fill_value=0.0)
    por_dia = por_dia.reindex(columns=range(1, 367), fill_value=0.0)
    acum = por_dia.cumsum(axis=1)

    ref_anios = [a for a in _anios_completos(s) if a != anio]
    if len(ref_anios) < MIN_ANIOS_REF or anio not in acum.index:
        raise ValueError("No hay suficientes años de referencia para comparar.")
    ref = acum.loc[ref_anios]

    corte = int(d.loc[d["anio"] == anio, "doy"].max())
    actual = acum.loc[anio, :corte]
    en_corte = ref[corte]
    lugar = 1 + int((en_corte > actual.iloc[-1]).sum())
    return {
        "actual": actual,
        "p10": ref.quantile(0.10), "p50": ref.quantile(0.50), "p90": ref.quantile(0.90),
        "corte_doy": corte,
        "lugar": lugar, "de": len(ref_anios) + 1,
        "percentil": 100 * float((en_corte <= actual.iloc[-1]).mean()),
        "vs_mediana_pct": 100 * (actual.iloc[-1] / ref[corte].median() - 1) if ref[corte].median() else np.nan,
    }


# ---------------------------------------------------------------- gráficas (devuelven figuras)
def _titulo_col(col):
    _validar_col(col)
    return "Celdas con fuego" if col == "n_arm" else "FRP total (MW)"


def fig_calendario(serie, col="n_arm", cmap="viridis"):
    """Calendario de actividad: año (filas) contra semana del año (columnas)."""
    _validar_col(col)
    s = preparar(serie)
    d = s[col].to_frame("v")
    d["anio"], d["sem"] = d.index.year, np.minimum((d.index.dayofyear - 1) // 7, 51)
    m = d.groupby(["anio", "sem"])["v"].sum().unstack()
    fig, ax = plt.subplots(figsize=(11, 0.28 * len(m) + 1.6))
    valores = m.to_numpy()
    vmax = float(np.nanmax(valores)) if valores.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    im = ax.imshow(valores, aspect="auto", cmap=cmap,
                   norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vmax))
    ax.set_yticks(range(len(m)), m.index)
    meses = ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    ax.set_xticks([round(i * 52 / 12) for i in range(12)], meses)
    ax.set_title(f"Calendario de actividad: {_titulo_col(col)}", loc="left")
    fig.colorbar(im, ax=ax, label="Total semanal", pad=0.01)
    fig.tight_layout()
    return fig


def fig_serie_diaria(serie, col="n_arm", ventana=7):
    """Serie diaria con media móvil: útil cuando hay pocos años para comparar."""
    _validar_col(col)
    v = preparar(serie)[col]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(v.index, v.to_numpy(), color=AZUL, alpha=0.55, width=1.0, label="Diario")
    ax.plot(v.index, v.rolling(ventana, center=True, min_periods=1).mean(), color=NARANJA, lw=2,
            label=f"Media de {ventana} días")
    ax.set_ylabel("Total diario")
    ax.set_title(f"Serie diaria: {_titulo_col(col)}", loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def fig_estacionalidad(est, col="n_arm"):
    _validar_col(col)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = est.index
    ax.bar(x, est["media"], color=AZUL, alpha=0.85, label="Promedio")
    ax.vlines(x, est["p10"], est["p90"], color=GRIS, lw=2, label="Rango p10 a p90")
    ax.set_xticks(x, ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("Total mensual")
    ax.set_title(f"Estacionalidad: {_titulo_col(col)}", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_anomalias(anom, anio, col="n_arm"):
    _validar_col(col)
    a = anom[anom["anio"] == anio].set_index("mes").reindex(range(1, 13))
    colores = np.where(a["z"].fillna(0) >= 0, NARANJA, AZUL)
    fig, ax = plt.subplots(figsize=(8, 4))
    presentes = a["z"].dropna()
    ax.bar(presentes.index, presentes, color=colores[presentes.index - 1])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(1, 13), ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("Desviaciones estándar vs. otros años")
    ax.set_title(f"Anomalía mensual {anio}: {_titulo_col(col)}", loc="left")
    fig.tight_layout()
    return fig


def fig_totales_anuales(anual, tend, col="n_arm"):
    _validar_col(col)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(anual.index, anual["total"], color=AZUL, alpha=0.85)
    if np.isfinite(tend["pendiente_por_anio"]):
        xs = np.array([anual.index.min(), anual.index.max()])
        ax.plot(xs, tend["ordenada"] + tend["pendiente_por_anio"] * xs, color=NARANJA, lw=2,
                label=f"Tendencia: {tend['cambio_pct_por_decada']:+.1f}% por década")
        ax.legend(frameon=False)
    ax.set_ylabel("Total anual")
    ax.set_title(f"Totales anuales: {_titulo_col(col)}", loc="left")
    fig.tight_layout()
    return fig


def fig_acumulado(res, anio, col="n_arm"):
    _validar_col(col)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = res["p50"].index
    ax.fill_between(x, res["p10"], res["p90"], color=GRIS, alpha=0.25, label="Años anteriores (p10 a p90)")
    ax.plot(x, res["p50"], color=GRIS, lw=1.5, label="Mediana de años anteriores")
    ax.plot(res["actual"].index, res["actual"], color=NARANJA, lw=2.5, label=str(anio))
    ax.set_xlabel("Día del año")
    ax.set_ylabel("Acumulado")
    ax.set_title(f"Temporada {anio}: {_titulo_col(col)}, lugar {res['lugar']} de {res['de']} a la fecha "
                 f"({res['vs_mediana_pct']:+.0f}% vs. mediana)", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- ejecución
def main(salida="figuras_demo", anio=2026, puntos=None, pais=None):
    if pais is not None and not puntos:
        raise ValueError("--pais requiere proporcionar --puntos.")

    out = Path(salida)
    out.mkdir(parents=True, exist_ok=True)

    if puntos:
        celdas = puntos_a_celdas(limpiar_puntos(leer_puntos(puntos)))
        if pais is None and "pais" in celdas.columns:
            print("Países con más actividad:\n", top_paises(celdas, 5).to_string(index=False))
        serie = serie_region(celdas, pais=pais)
        anio = int(serie["fecha"].max().year)
    else:
        serie = generar_serie_falsa()

    n_anios = len(_anios_completos(preparar(serie)))
    print(f"\nDías: {len(serie)} | años completos: {n_anios}")
    fig_calendario(serie).savefig(out / "calendario.png", dpi=130)
    fig_serie_diaria(serie).savefig(out / "serie_diaria.png", dpi=130)

    if n_anios < MIN_ANIOS_REF:
        print(f"Con {n_anios} año(s) completo(s) no se calculan estacionalidad, tendencia ni anomalías "
              f"(se necesitan al menos {MIN_ANIOS_REF}).")
        mensual = _meses_completos(preparar(serie))
        print("\nTotal mensual de celdas con fuego:\n", mensual["n_arm"].round(0).astype(int).to_string())
        print("\nFiguras en", out.resolve())
        return

    est = estacionalidad(serie)
    anual, tend = totales_anuales(serie)
    anom = anomalias_mensuales(serie)
    acum = acumulado_temporada(serie, anio)

    print("Estacionalidad (media mensual):\n", est["media"].round(0).to_string())
    print("\nTendencia:", {k: round(float(v), 2) for k, v in tend.items()})
    print(f"\nTemporada {anio}: lugar {acum['lugar']} de {acum['de']}, "
          f"percentil {acum['percentil']:.0f}, {acum['vs_mediana_pct']:+.0f}% vs. mediana")

    fig_estacionalidad(est).savefig(out / "estacionalidad.png", dpi=130)
    fig_anomalias(anom, anio - 1).savefig(out / "anomalias.png", dpi=130)
    fig_totales_anuales(anual, tend).savefig(out / "totales_anuales.png", dpi=130)
    fig_acumulado(acum, anio).savefig(out / "acumulado.png", dpi=130)
    anom.to_csv(out / "anomalias_mensuales.csv", index=False)
    print("\nFiguras en", out.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Análisis de patrones de incendios.")
    ap.add_argument("--puntos", help="parquet o CSV de puntos (si se omite, usa datos simulados)")
    ap.add_argument("--pais", help="nombre del país tal como viene en la columna country")
    ap.add_argument("--salida", default="figuras_demo")
    a = ap.parse_args()
    if a.pais is not None and a.puntos is None:
        ap.error("--pais requiere proporcionar --puntos.")
    main(salida=a.salida, puntos=a.puntos, pais=a.pais)
