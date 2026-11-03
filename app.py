import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="¿Qué tan popular suena?", layout="wide")

VERDE, NEGRO, GRIS = "#1DB954", "#191414", "#8A8A8A"

st.markdown(
    f"""
    <style>
    h1, h2, h3 {{ letter-spacing: -0.5px; }}
    .stMetric {{ background: rgba(29,185,84,0.08); padding: 12px 16px; border-radius: 12px; }}
    div.stButton > button {{ border-radius: 20px; border: 1px solid {VERDE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar():
    modelo = joblib.load("modelo_rf.joblib")
    with open("meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return modelo, meta


modelo, meta = cargar()
FEATS = meta["features"]
R = meta["rangos"]
MEDIA = float(meta["media_popularidad"])

ETIQUETAS = {
    "danceability": ("Bailabilidad", "Qué tan fácil es bailar la canción (0 a 1)"),
    "energy": ("Energía", "Intensidad y actividad de la canción (0 a 1)"),
    "loudness": ("Volumen (dB)", "Volumen promedio; más cercano a 0 = más fuerte"),
    "tempo": ("Tempo (BPM)", "Velocidad de la canción, pulsos por minuto"),
    "instrumentalness": ("Instrumentalidad", "Cercano a 1 = sin voz, solo instrumentos"),
}


def etiqueta(f):
    return ETIQUETAS.get(f, (f, ""))[0]


# ---------- Estado inicial de los sliders ----------
for f in FEATS:
    st.session_state.setdefault(f"v_{f}", float(R[f]["median"]))


def aplicar(valores):
    for f in FEATS:
        st.session_state[f"v_{f}"] = float(valores[f])


def preset_promedio():
    aplicar({f: R[f]["median"] for f in FEATS})


def preset_bailable():
    v = {f: R[f]["median"] for f in FEATS}
    for f in ("danceability", "energy", "loudness"):
        if f in v:
            v[f] = R[f]["p95"]
    if "instrumentalness" in v:
        v["instrumentalness"] = R["instrumentalness"]["p5"]
    aplicar(v)


def preset_tranquila():
    v = {f: R[f]["median"] for f in FEATS}
    for f in ("danceability", "energy", "loudness", "tempo"):
        if f in v:
            v[f] = R[f]["p5"]
    aplicar(v)


def preset_instrumental():
    v = {f: R[f]["median"] for f in FEATS}
    if "instrumentalness" in v:
        v["instrumentalness"] = R["instrumentalness"]["p95"]
    aplicar(v)


# ---------- Barra lateral ----------
st.sidebar.title("Características de la canción")
for f in FEATS:
    lo, hi = float(R[f]["min"]), float(R[f]["max"])
    st.sidebar.slider(
        etiqueta(f),
        min_value=lo,
        max_value=hi,
        step=(hi - lo) / 200,
        key=f"v_{f}",
        help=ETIQUETAS.get(f, ("", ""))[1],
    )

st.sidebar.markdown("**Perfiles rápidos**")
c1, c2 = st.sidebar.columns(2)
c1.button("Bailable", on_click=preset_bailable, width="stretch")
c2.button("Tranquila", on_click=preset_tranquila, width="stretch")
c3, c4 = st.sidebar.columns(2)
c3.button("Instrumental", on_click=preset_instrumental, width="stretch")
c4.button("Promedio", on_click=preset_promedio, width="stretch")


def valores_actuales():
    return {f: st.session_state[f"v_{f}"] for f in FEATS}


def predecir(df):
    return modelo.predict(df[FEATS])


actual = pd.DataFrame([valores_actuales()])
pred = float(predecir(actual)[0])
rmse = float(meta["rmse_test"])

# ---------- Encabezado ----------
st.title("¿Qué tan popular suena una canción?")
st.caption(
    "Simulador basado en un Random Forest entrenado solo con características del sonido. "
    "Mueve los controles de la izquierda y observa cómo cambia la popularidad estimada (0 a 100)."
)

tab1, tab2, tab3 = st.tabs(["Predicción", "¿Qué pasa si muevo una variable?", "¿Qué tan confiable es?"])

# ---------- Pestaña 1 ----------
with tab1:
    izq, der = st.columns([1.2, 1])
    with izq:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                number={"valueformat": ".1f"},
                delta={"reference": MEDIA, "valueformat": "+.1f", "suffix": " vs. promedio"},
                title={"text": "Popularidad estimada (0 a 100)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": VERDE},
                    "steps": [{"range": [max(0, pred - rmse), min(100, pred + rmse)], "color": "#E3F5EA"}],
                    "threshold": {"line": {"color": NEGRO, "width": 3}, "value": MEDIA},
                },
            )
        )
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=70, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "La línea negra es la popularidad promedio del dataset. La franja verde clara es el margen de error "
            f"típico del modelo (±{rmse:.0f} puntos)."
        )

    with der:
        st.metric("Popularidad estimada", f"{pred:.1f} / 100", f"{pred - MEDIA:+.1f} vs. promedio ({MEDIA:.1f})")
        st.metric("Margen de error típico del modelo", f"± {rmse:.1f} puntos")
        st.markdown("**Valores actuales**")
        st.dataframe(
            pd.DataFrame({"Variable": [etiqueta(f) for f in FEATS], "Valor": [round(actual[f][0], 3) for f in FEATS]}),
            hide_index=True,
            width="stretch",
        )

    if abs(pred - MEDIA) < 0.25 * rmse:
        st.info(
            "Fíjate: aunque muevas todos los controles, la estimación casi no se separa del promedio y el margen "
            "de error es mucho mayor que cualquier cambio. No es un error de la app: indica que, en este dataset, "
            "el sonido por sí solo dice muy poco sobre la popularidad."
        )

# ---------- Pestaña 2 ----------
with tab2:
    var = st.selectbox("Variable a recorrer", FEATS, format_func=etiqueta)
    lo, hi = float(R[var]["min"]), float(R[var]["max"])
    malla = np.linspace(lo, hi, 80)
    barrido = pd.DataFrame([valores_actuales()] * len(malla))
    barrido[var] = malla
    curva = predecir(barrido)

    zoom = st.checkbox("Acercar el eje vertical (exagera las diferencias)", value=False)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=malla, y=curva, mode="lines", line=dict(color=VERDE, width=3), name="Popularidad estimada"))
    fig2.add_trace(
        go.Scatter(x=[actual[var][0]], y=[pred], mode="markers", marker=dict(size=14, color=NEGRO), name="Canción actual")
    )
    fig2.add_hline(y=MEDIA, line_dash="dash", line_color=GRIS, annotation_text="Promedio del dataset")
    fig2.update_layout(
        height=420,
        xaxis_title=etiqueta(var),
        yaxis_title="Popularidad estimada",
        yaxis=dict(range=None if zoom else [0, 100]),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig2, width="stretch")
    st.caption(
        f"Al recorrer **{etiqueta(var)}** de {lo:.2f} a {hi:.2f}, la popularidad estimada varía entre "
        f"**{curva.min():.1f}** y **{curva.max():.1f}** (una diferencia de {curva.max() - curva.min():.1f} puntos), "
        f"frente a un margen de error típico de ±{rmse:.0f} puntos."
    )

# ---------- Pestaña 3 ----------
with tab3:
    st.subheader("Transparencia del modelo (evaluado con canciones que no vio al entrenar)")
    a, b, c = st.columns(3)
    a.metric("R² (0 = igual que el promedio)", f"{meta['r2_test']:.4f}")
    b.metric("Error típico (RMSE)", f"{rmse:.2f}", f"{rmse - meta['rmse_base']:+.2f} vs. adivinar el promedio", delta_color="inverse")
    c.metric("Error absoluto medio (MAE)", f"{meta['mae_test']:.2f}", f"{meta['mae_test'] - meta['mae_base']:+.2f} vs. adivinar el promedio", delta_color="inverse")

    st.markdown(
        "- **R²**: qué parte de las diferencias de popularidad logra explicar el modelo. 0 = nada; 1 = todo.\n"
        "- **RMSE y MAE**: cuántos puntos se equivoca el modelo, en promedio. Si es igual al error de siempre decir el "
        "promedio, el modelo no aporta."
    )

    imp = pd.Series(meta["perm_importance"]).sort_values()
    fig3 = go.Figure(go.Bar(x=imp.values, y=[etiqueta(i) for i in imp.index], orientation="h", marker_color=VERDE))
    fig3.update_layout(
        height=320,
        title="Importancia por permutación: cuánto empeora el modelo (en R²) si se desordena cada variable",
        xaxis_title="Pérdida de R² (cerca de 0 = no aporta)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig3, width="stretch")

    st.warning(
        "**Limitación importante:** la popularidad real depende de factores que este dataset no contiene "
        "(qué tan conocido es el artista, promoción, si la canción es coherente con su estilo anterior, tendencias en redes). "
        "Por eso este simulador muestra *tendencias del sonido*, no una predicción confiable de éxito."
    )
