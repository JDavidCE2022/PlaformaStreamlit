# -*- coding: utf-8 -*-
"""
Simulador y Analizador de Temperatura - Medellín, Colombia
============================================================
Aplicación Streamlit que:
  1. Simula datos de temperatura de la ciudad de Medellín (serie horaria o diaria).
  2. Presenta análisis cuantitativo (estadística descriptiva) y cualitativo (narrativa
     interpretativa automática) de los datos simulados.
  3. Ofrece gráficos interactivos y controles dinámicos para explorar los resultados.

NOTA: Los datos son SIMULADOS con un modelo estadístico simplificado inspirado en el
clima real de Medellín ("ciudad de la eterna primavera": temperatura media ~22-23°C,
poca variación estacional, ciclo diario marcado). No provienen de una fuente
meteorológica oficial.

Para ejecutar:
    pip install -r requirements.txt
    streamlit run app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Clima Medellín · Simulador",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Parámetros climáticos de referencia para Medellín (aprox.)
TEMP_MEDIA_ANUAL = 22.5   # °C, promedio histórico aproximado
AMPLITUD_ESTACIONAL = 1.5  # °C, la variación estación seca/lluviosa es leve
AMPLITUD_DIARIA = 6.5      # °C, diferencia entre el pico de la tarde y la madrugada
HORA_PICO = 14             # hora del día con temperatura máxima aprox.

UMBRAL_FRIO = 18.0
UMBRAL_CALIDO = 25.0

MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}


# ----------------------------------------------------------------------------
# SIMULACIÓN DE DATOS
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def simular_temperatura(fecha_inicio: dt.date, dias: int, resolucion: str, semilla: int) -> pd.DataFrame:
    """Genera una serie sintética de temperatura para Medellín.

    Componentes del modelo:
      - Nivel base (temperatura media anual).
      - Componente estacional suave (seno de baja frecuencia sobre el día del año).
      - Ciclo diario (seno de alta frecuencia, solo si la resolución es horaria).
      - Ruido aleatorio gaussiano.
      - Eventos ocasionales (frentes fríos / olas de calor de corta duración).
    """
    rng = np.random.default_rng(semilla)

    freq = "h" if resolucion == "Horaria" else "D"
    periodos = dias * 24 if resolucion == "Horaria" else dias
    fechas = pd.date_range(start=fecha_inicio, periods=periodos, freq=freq)

    dia_del_anio = fechas.dayofyear.values
    estacional = AMPLITUD_ESTACIONAL * np.sin(2 * np.pi * (dia_del_anio - 80) / 365.0)

    if resolucion == "Horaria":
        hora = fechas.hour.values + fechas.minute.values / 60.0
        ciclo_diario = AMPLITUD_DIARIA * np.sin(2 * np.pi * (hora - (HORA_PICO - 6)) / 24.0)
    else:
        ciclo_diario = np.zeros(periodos)

    ruido = rng.normal(0, 0.9, size=periodos)

    # Eventos ocasionales: frentes fríos (-) u olas de calor (+)
    eventos = np.zeros(periodos)
    unidad_evento = 24 if resolucion == "Horaria" else 1
    n_eventos = max(1, periodos // (unidad_evento * 12))  # ~ cada 12 días
    idx_eventos = rng.choice(periodos, size=min(n_eventos, periodos), replace=False)
    for idx in idx_eventos:
        magnitud = rng.choice([-1, 1]) * rng.uniform(1.5, 3.5)
        duracion = int(rng.integers(1, 4) * unidad_evento)
        fin = min(periodos, idx + duracion)
        eventos[idx:fin] += magnitud

    temperatura = TEMP_MEDIA_ANUAL + estacional + ciclo_diario + ruido + eventos

    df = pd.DataFrame({"fecha": fechas, "temperatura": np.round(temperatura, 2)})
    df["fecha_dia"] = df["fecha"].dt.date
    df["hora"] = df["fecha"].dt.hour
    df["mes"] = df["fecha"].dt.month
    df["mes_nombre"] = df["mes"].map(MESES_ES)
    df["dia_semana"] = df["fecha"].dt.day_name()

    def clasificar(t):
        if t < UMBRAL_FRIO:
            return "Frío"
        elif t > UMBRAL_CALIDO:
            return "Cálido"
        return "Templado/Agradable"

    df["condicion"] = df["temperatura"].apply(clasificar)
    return df


# ----------------------------------------------------------------------------
# BARRA LATERAL: PARÁMETROS DE SIMULACIÓN
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Parámetros de simulación")

if "semilla" not in st.session_state:
    st.session_state.semilla = 42

fecha_inicio = st.sidebar.date_input(
    "Fecha de inicio",
    value=dt.date(2025, 1, 1),
    help="Fecha a partir de la cual se generan los datos simulados.",
)

dias = st.sidebar.slider(
    "Días a simular",
    min_value=7,
    max_value=365,
    value=90,
    step=1,
)

resolucion = st.sidebar.radio(
    "Resolución temporal",
    options=["Horaria", "Diaria"],
    index=0,
    help="Horaria genera 24 registros por día (recomendado para ver el ciclo diario).",
)

semilla = st.sidebar.number_input(
    "Semilla aleatoria",
    min_value=0,
    max_value=999_999,
    value=st.session_state.semilla,
    step=1,
)
st.session_state.semilla = semilla

if st.sidebar.button("🎲 Nueva simulación aleatoria"):
    st.session_state.semilla = int(np.random.default_rng().integers(0, 999_999))
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Los datos son **simulados** con un modelo estadístico simplificado inspirado "
    "en el clima real de Medellín. No corresponden a mediciones oficiales."
)

# Generar datos
df = simular_temperatura(fecha_inicio, dias, resolucion, st.session_state.semilla)

# ----------------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------------
st.title("🌤️ Simulador de Temperatura — Medellín, Colombia")
st.markdown(
    f"Serie simulada de **{len(df):,}** registros · Resolución **{resolucion.lower()}** · "
    f"Del **{df['fecha'].min().strftime('%d/%m/%Y')}** al **{df['fecha'].max().strftime('%d/%m/%Y')}**"
)

tab_resumen, tab_cuant, tab_cual, tab_graf, tab_datos = st.tabs(
    ["📊 Resumen", "🔢 Análisis Cuantitativo", "📝 Análisis Cualitativo", "📈 Gráficos", "🔍 Explorar Datos"]
)

# ----------------------------------------------------------------------------
# TAB 1 — RESUMEN
# ----------------------------------------------------------------------------
with tab_resumen:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperatura media", f"{df['temperatura'].mean():.1f} °C")
    col2.metric("Máxima registrada", f"{df['temperatura'].max():.1f} °C")
    col3.metric("Mínima registrada", f"{df['temperatura'].min():.1f} °C")
    col4.metric("Desviación estándar", f"{df['temperatura'].std():.2f} °C")

    st.markdown("#### Vista previa de la serie temporal")
    fig_preview = px.line(
        df, x="fecha", y="temperatura",
        labels={"fecha": "Fecha", "temperatura": "Temperatura (°C)"},
    )
    fig_preview.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    fig_preview.add_hline(y=df["temperatura"].mean(), line_dash="dot", line_color="gray",
                           annotation_text="Media", annotation_position="top left")
    st.plotly_chart(fig_preview, use_container_width=True)

    with st.expander("ℹ️ ¿Cómo se generan estos datos?"):
        st.markdown(
            """
            El modelo de simulación combina:
            - **Nivel base**: temperatura media anual de referencia (~22.5 °C).
            - **Componente estacional**: variación leve asociada a épocas seca/lluviosa.
            - **Ciclo diario**: oscilación entre la madrugada (más fría) y la tarde (más cálida),
              visible solo con resolución horaria.
            - **Ruido aleatorio**: variabilidad natural día a día.
            - **Eventos ocasionales**: frentes fríos u olas de calor de corta duración.
            """
        )

# ----------------------------------------------------------------------------
# TAB 2 — ANÁLISIS CUANTITATIVO
# ----------------------------------------------------------------------------
with tab_cuant:
    st.markdown("### Estadística descriptiva")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(
            df["temperatura"].describe().rename("valor").to_frame().style.format("{:.2f}"),
            use_container_width=True,
        )
    with c2:
        percentiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        pct_df = df["temperatura"].quantile(percentiles).rename("temperatura (°C)").to_frame()
        pct_df.index = [f"P{int(p * 100)}" for p in percentiles]
        st.dataframe(pct_df.style.format("{:.2f}"), use_container_width=True)

    st.markdown("### Promedio mensual")
    resumen_mensual = (
        df.groupby(["mes", "mes_nombre"])["temperatura"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values("mes")
        .rename(columns={"mean": "media", "std": "desv_std", "min": "mínima", "max": "máxima", "count": "n"})
    )
    st.dataframe(
        resumen_mensual.drop(columns="mes").set_index("mes_nombre").style.format(
            {"media": "{:.2f}", "desv_std": "{:.2f}", "mínima": "{:.2f}", "máxima": "{:.2f}"}
        ),
        use_container_width=True,
    )

    if resolucion == "Horaria":
        st.markdown("### Correlación hora del día vs. temperatura")
        corr = df["hora"].corr(df["temperatura"])
        st.metric("Coeficiente de correlación (Pearson)", f"{corr:.2f}")
        st.caption(
            "Un valor cercano a 0 no implica ausencia de relación: la temperatura sigue un patrón "
            "cíclico (senoidal), no lineal, respecto a la hora del día."
        )

    st.markdown("### Media móvil")
    ventana = st.slider(
        "Tamaño de ventana para la media móvil (registros)",
        min_value=2,
        max_value=max(3, min(200, len(df) // 2)),
        value=min(24, max(2, len(df) // 10)),
    )
    df_mm = df.copy()
    df_mm["media_movil"] = df_mm["temperatura"].rolling(window=ventana, min_periods=1).mean()
    fig_mm = go.Figure()
    fig_mm.add_trace(go.Scatter(x=df_mm["fecha"], y=df_mm["temperatura"], name="Temperatura", opacity=0.35))
    fig_mm.add_trace(go.Scatter(x=df_mm["fecha"], y=df_mm["media_movil"], name=f"Media móvil ({ventana})", line=dict(width=3)))
    fig_mm.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Temperatura (°C)", xaxis_title="Fecha")
    st.plotly_chart(fig_mm, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — ANÁLISIS CUALITATIVO
# ----------------------------------------------------------------------------
with tab_cual:
    st.markdown("### Interpretación automática de los datos")

    media = df["temperatura"].mean()
    desv = df["temperatura"].std()
    t_max = df["temperatura"].max()
    fecha_max = df.loc[df["temperatura"].idxmax(), "fecha"]
    t_min = df["temperatura"].min()
    fecha_min = df.loc[df["temperatura"].idxmin(), "fecha"]

    conteo = df["condicion"].value_counts(normalize=True).mul(100).round(1)
    pct_frio = conteo.get("Frío", 0.0)
    pct_calido = conteo.get("Cálido", 0.0)
    pct_templado = conteo.get("Templado/Agradable", 0.0)

    if desv < 1.5:
        estabilidad = "muy estable, con oscilaciones mínimas día a día"
    elif desv < 3:
        estabilidad = "moderadamente estable, típica del clima ecuatorial de montaña"
    else:
        estabilidad = "más variable de lo habitual para la región"

    narrativa = f"""
Durante el periodo simulado, la temperatura promedio fue de **{media:.1f} °C**, con una
desviación estándar de **{desv:.2f} °C**, lo que indica un comportamiento **{estabilidad}**.
Esto es consistente con la fama de Medellín como *"ciudad de la eterna primavera"*, dado que
la ciudad no presenta estaciones marcadas como frío o verano extremos.

El registro más alto fue de **{t_max:.1f} °C** el **{fecha_max.strftime('%d/%m/%Y %H:%M')}**,
mientras que el más bajo fue de **{t_min:.1f} °C** el **{fecha_min.strftime('%d/%m/%Y %H:%M')}**.

Clasificando cada registro según su temperatura:
- 🥶 **Frío** (< {UMBRAL_FRIO:.0f} °C): {pct_frio:.1f}% del tiempo.
- 😊 **Templado/Agradable** ({UMBRAL_FRIO:.0f}–{UMBRAL_CALIDO:.0f} °C): {pct_templado:.1f}% del tiempo.
- 🥵 **Cálido** (> {UMBRAL_CALIDO:.0f} °C): {pct_calido:.1f}% del tiempo.
"""
    st.markdown(narrativa)

    if pct_templado >= 60:
        st.success(
            "✅ La mayor parte del tiempo simulado se mantiene en un rango **agradable**, "
            "coherente con el clima típico de Medellín."
        )
    elif pct_calido > pct_frio:
        st.warning(
            "⚠️ Predominan las condiciones **cálidas** en esta simulación. Podría recomendarse "
            "mayor hidratación y protección solar en actividades al aire libre."
        )
    else:
        st.info(
            "ℹ️ Predominan las condiciones **frías** en esta simulación. Podría recomendarse "
            "abrigo ligero, especialmente en horas de la madrugada."
        )

    st.markdown("### Distribución de condiciones")
    fig_cond = px.pie(
        conteo.rename("porcentaje").reset_index().rename(columns={"index": "condicion"}),
        names="condicion", values="porcentaje", hole=0.45,
        color="condicion",
        color_discrete_map={"Frío": "#4C9BE8", "Templado/Agradable": "#59C46B", "Cálido": "#E86B4C"},
    )
    fig_cond.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_cond, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4 — GRÁFICOS INTERACTIVOS
# ----------------------------------------------------------------------------
with tab_graf:
    st.markdown("### Serie temporal completa")
    fig_serie = px.line(df, x="fecha", y="temperatura", labels={"fecha": "Fecha", "temperatura": "Temperatura (°C)"})
    fig_serie.update_xaxes(rangeslider_visible=True)
    fig_serie.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_serie, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Distribución (histograma)")
        n_bins = st.slider("Número de bins", min_value=5, max_value=60, value=25)
        fig_hist = px.histogram(df, x="temperatura", nbins=n_bins, labels={"temperatura": "Temperatura (°C)"})
        fig_hist.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        st.markdown("### Boxplot por mes")
        fig_box = px.box(
            df.sort_values("mes"), x="mes_nombre", y="temperatura",
            labels={"mes_nombre": "Mes", "temperatura": "Temperatura (°C)"},
        )
        fig_box.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

    if resolucion == "Horaria":
        st.markdown("### Mapa de calor: temperatura promedio por hora y mes")
        pivot = df.pivot_table(index="hora", columns="mes_nombre", values="temperatura", aggfunc="mean")
        orden_meses = [MESES_ES[m] for m in sorted(df["mes"].unique())]
        pivot = pivot[orden_meses]
        fig_heat = px.imshow(
            pivot, aspect="auto", color_continuous_scale="RdYlBu_r",
            labels=dict(x="Mes", y="Hora del día", color="°C"),
        )
        fig_heat.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.caption("💡 Cambia la resolución a 'Horaria' en la barra lateral para ver el mapa de calor hora × mes.")

# ----------------------------------------------------------------------------
# TAB 5 — EXPLORAR DATOS
# ----------------------------------------------------------------------------
with tab_datos:
    st.markdown("### Filtros dinámicos")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        rango_fechas = st.date_input(
            "Rango de fechas",
            value=(df["fecha_dia"].min(), df["fecha_dia"].max()),
            min_value=df["fecha_dia"].min(),
            max_value=df["fecha_dia"].max(),
        )
    with col_f2:
        rango_temp = st.slider(
            "Rango de temperatura (°C)",
            float(df["temperatura"].min()), float(df["temperatura"].max()),
            (float(df["temperatura"].min()), float(df["temperatura"].max())),
        )

    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        f_ini, f_fin = rango_fechas
    else:
        f_ini, f_fin = df["fecha_dia"].min(), df["fecha_dia"].max()

    df_filtrado = df[
        (df["fecha_dia"] >= f_ini) & (df["fecha_dia"] <= f_fin) &
        (df["temperatura"] >= rango_temp[0]) & (df["temperatura"] <= rango_temp[1])
    ]

    st.caption(f"Mostrando **{len(df_filtrado):,}** de **{len(df):,}** registros.")
    st.dataframe(
        df_filtrado[["fecha", "temperatura", "condicion", "mes_nombre", "dia_semana"]],
        use_container_width=True, height=400,
    )

    csv = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        data=csv,
        file_name="temperatura_medellin_simulada.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Simulador educativo · Datos sintéticos · Creado con Streamlit + Plotly")
