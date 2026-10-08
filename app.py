import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Predicción de abandono de empleados",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def cargar_datos():
    return joblib.load("modelos_despliegue.joblib")

try:
    data = cargar_datos()
except FileNotFoundError:
    st.error(
        "No se encontró modelos_despliegue.joblib. "
        "Ejecute primero la celda de exportación del notebook."
    )
    st.stop()

modelos = data["models"]
columnas_modelo = data["columns"]
metricas = pd.DataFrame(data["metrics"])
matrices = data["confusion_matrices"]
roc_data = data["roc_data"]
pr_data = data["pr_data"]
importancias = data.get("feature_importance", {})
coeficientes = data.get("coefficients", {})
tiempos = pd.DataFrame(data.get("optimization_times", []))
categoricas = data.get("categorical_values", {})
defaults = data.get("numeric_defaults", {})
mins = data.get("numeric_min", {})
maxs = data.get("numeric_max", {})
pca_data = pd.DataFrame(data.get("pca_data", []))
pca_variance = data.get("pca_variance", [])

# ============================================================
# ESTILO
# ============================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    letter-spacing: -0.02em;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem;
}
.small-note {
    color: #6b7280;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Navegación")
pagina = st.sidebar.radio(
    "Seleccione una sección",
    [
        "Resumen",
        "Comparación de modelos",
        "Análisis del modelo",
        "Predicción individual"
    ]
)

st.sidebar.divider()

modelo_seleccionado = st.sidebar.selectbox(
    "Modelo para análisis",
    list(modelos.keys()),
    index=(
        list(modelos.keys()).index("Random Forest")
        if "Random Forest" in modelos else 0
    )
)

st.sidebar.caption(
    "El modelo final del estudio es Random Forest. "
    "La interfaz permite consultar también las alternativas evaluadas."
)

# ============================================================
# ENCABEZADO
# ============================================================
st.title("Predicción del abandono de empleados")
st.caption(
    "Aplicación interactiva desarrollada a partir de los modelos "
    "evaluados en la ACA 2."
)

# ============================================================
# RESUMEN
# ============================================================
if pagina == "Resumen":

    st.header("Resumen del estudio")

    mejor_recall = metricas.loc[metricas["Recall"].idxmax()]
    mejor_f1 = metricas.loc[metricas["F1"].idxmax()]
    mejor_auc = metricas.loc[metricas["AUC"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Modelos evaluados",
            len(modelos)
        )

    with c2:
        st.metric(
            "Mayor Recall",
            mejor_recall["Modelo"],
            f'{mejor_recall["Recall"]:.4f}'
        )

    with c3:
        st.metric(
            "Mayor F1",
            mejor_f1["Modelo"],
            f'{mejor_f1["F1"]:.4f}'
        )

    with c4:
        st.metric(
            "Mayor ROC-AUC",
            mejor_auc["Modelo"],
            f'{mejor_auc["AUC"]:.4f}'
        )

    st.info(
        "Para este estudio, la clase de interés es «Abandonó». "
        "Por ello, Recall y F1-score de esta clase tienen especial "
        "importancia para la selección del modelo."
    )

    st.subheader("Resultados principales")

    columnas_mostrar = [
        "Modelo", "Accuracy", "Precision",
        "Recall", "F1", "F1_Macro", "AUC", "PR_AUC"
    ]

    st.dataframe(
        metricas[columnas_mostrar].style.format({
            c: "{:.4f}" for c in columnas_mostrar
            if c != "Modelo"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Modelo seleccionado")

    if "Random Forest" in modelos:
        rf = metricas[
            metricas["Modelo"] == "Random Forest"
        ].iloc[0]

        st.success(
            f"Random Forest fue seleccionado en el estudio por su "
            f"desempeño en la identificación de la clase «Abandonó»: "
            f"Recall = {rf['Recall']:.4f} y F1 = {rf['F1']:.4f}."
        )

    st.warning(
        "Los resultados son de apoyo analítico. Un caso clasificado "
        "como posible abandono no constituye una determinación causal "
        "ni una decisión automática sobre un empleado."
    )

# ============================================================
# COMPARACIÓN
# ============================================================
elif pagina == "Comparación de modelos":

    st.header("Comparación de modelos")

    st.write(
        "La comparación se realiza sobre el conjunto de prueba utilizando "
        "las métricas calculadas en el notebook."
    )

    tab1, tab2, tab3 = st.tabs([
        "Métricas",
        "Errores de clasificación",
        "Costo computacional"
    ])

    with tab1:
        columnas = [
            "Modelo", "Accuracy", "Precision",
            "Recall", "F1", "F1_Macro",
            "AUC", "PR_AUC"
        ]

        st.dataframe(
            metricas[columnas].style.format({
                c: "{:.4f}" for c in columnas if c != "Modelo"
            }),
            use_container_width=True,
            hide_index=True
        )

        metricas_largas = metricas.melt(
            id_vars="Modelo",
            value_vars=[
                "Accuracy", "Precision", "Recall",
                "F1", "F1_Macro", "AUC", "PR_AUC"
            ],
            var_name="Métrica",
            value_name="Valor"
        )

        fig = px.bar(
            metricas_largas,
            x="Modelo",
            y="Valor",
            color="Métrica",
            barmode="group",
            title="Comparación de métricas"
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        f1 = metricas[
            ["Modelo", "F1"]
        ].sort_values("F1", ascending=True)

        fig_f1 = px.bar(
            f1,
            x="F1",
            y="Modelo",
            orientation="h",
            text="F1",
            title="F1-score para la clase Abandonó"
        )
        fig_f1.update_xaxes(range=[0, max(0.5, f1["F1"].max() + 0.08)])
        st.plotly_chart(fig_f1, use_container_width=True)

    with tab2:
        tabla_error = []

        for nombre in modelos:
            cm = np.array(matrices[nombre])
            tn, fp, fn, tp = cm.ravel()

            tabla_error.append({
                "Modelo": nombre,
                "Verdaderos positivos": int(tp),
                "Falsos negativos": int(fn),
                "Falsos positivos": int(fp),
                "Verdaderos negativos": int(tn)
            })

        df_error = pd.DataFrame(tabla_error)

        st.dataframe(
            df_error,
            use_container_width=True,
            hide_index=True
        )

        fig_error = px.bar(
            df_error,
            x="Modelo",
            y=["Verdaderos positivos", "Falsos negativos"],
            barmode="group",
            title="Verdaderos positivos y falsos negativos"
        )
        st.plotly_chart(fig_error, use_container_width=True)

        st.caption(
            "En este problema los falsos negativos son especialmente "
            "relevantes porque corresponden a casos reales de abandono "
            "que el modelo no identifica."
        )

    with tab3:
        if not tiempos.empty:
            fig_t = px.bar(
                tiempos.sort_values("Tiempo_segundos"),
                x="Modelo",
                y="Tiempo_segundos",
                text="Tiempo_segundos",
                title="Tiempo de optimización de hiperparámetros"
            )
            fig_t.update_traces(
                texttemplate="%{text:.3f} s",
                textposition="outside"
            )
            st.plotly_chart(fig_t, use_container_width=True)

            st.dataframe(
                tiempos.sort_values("Tiempo_segundos").style.format(
                    {"Tiempo_segundos": "{:.3f}"}
                ),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "El tiempo de optimización es un criterio complementario; "
                "no es el criterio principal de selección del modelo."
            )

# ============================================================
# ANÁLISIS DEL MODELO
# ============================================================
elif pagina == "Análisis del modelo":

    st.header(f"Análisis: {modelo_seleccionado}")

    fila = metricas[
        metricas["Modelo"] == modelo_seleccionado
    ].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Accuracy", f'{fila["Accuracy"]:.4f}')
    with c2:
        st.metric("Precision", f'{fila["Precision"]:.4f}')
    with c3:
        st.metric("Recall", f'{fila["Recall"]:.4f}')
    with c4:
        st.metric("F1", f'{fila["F1"]:.4f}')
    with c5:
        st.metric("ROC-AUC", f'{fila["AUC"]:.4f}')

    tab1, tab2, tab3, tab4 = st.tabs([
        "Matriz de confusión",
        "ROC y Precision-Recall",
        "Variables",
        "PCA"
    ])

    with tab1:
        cm = np.array(matrices[modelo_seleccionado])

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["No abandonó", "Abandonó"],
            y=["No abandonó", "Abandonó"],
            labels={
                "x": "Predicción",
                "y": "Valor real",
                "color": "Cantidad"
            },
            title=f"Matriz de confusión – {modelo_seleccionado}"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        tn, fp, fn, tp = cm.ravel()

        a, b, c, d = st.columns(4)
        with a:
            st.metric("Verdaderos positivos", int(tp))
        with b:
            st.metric("Falsos negativos", int(fn))
        with c:
            st.metric("Falsos positivos", int(fp))
        with d:
            st.metric("Verdaderos negativos", int(tn))

        st.warning(
            f"El modelo deja {int(fn)} casos reales de abandono "
            "clasificados como No abandonó."
        )

    with tab2:
        roc = roc_data[modelo_seleccionado]

        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=roc["fpr"],
                y=roc["tpr"],
                mode="lines",
                name=f"AUC = {fila['AUC']:.4f}"
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Referencia",
                line=dict(dash="dash")
            )
        )
        fig_roc.update_layout(
            title="Curva ROC",
            xaxis_title="Tasa de falsos positivos",
            yaxis_title="Tasa de verdaderos positivos"
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        pr = pr_data[modelo_seleccionado]

        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=pr["recall"],
                y=pr["precision"],
                mode="lines",
                name=f"PR-AUC = {fila['PR_AUC']:.4f}"
            )
        )
        fig_pr.update_layout(
            title="Curva Precision-Recall",
            xaxis_title="Recall",
            yaxis_title="Precision"
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    with tab3:
        if modelo_seleccionado in importancias:
            df_imp = pd.DataFrame(
                importancias[modelo_seleccionado]
            ).sort_values("Importancia")

            fig_imp = px.bar(
                df_imp,
                x="Importancia",
                y="Variable",
                orientation="h",
                title="Importancia de variables"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            st.caption(
                "La importancia predictiva no implica causalidad."
            )

        elif modelo_seleccionado in coeficientes:
            df_coef = pd.DataFrame(
                coeficientes[modelo_seleccionado]
            )
            df_coef["Abs"] = df_coef["Coeficiente"].abs()
            df_coef = df_coef.sort_values("Abs")

            fig_coef = px.bar(
                df_coef,
                x="Coeficiente",
                y="Variable",
                orientation="h",
                title="Coeficientes de Regresión Logística"
            )
            st.plotly_chart(fig_coef, use_container_width=True)

            st.caption(
                "El signo del coeficiente indica dirección dentro del "
                "modelo; no debe interpretarse como causalidad."
            )
        else:
            st.info(
                "Este modelo no cuenta con una medida directa de "
                "importancia de variables exportada."
            )

    with tab4:
        if not pca_data.empty:
            fig_pca = px.scatter_3d(
                pca_data,
                x="PC1",
                y="PC2",
                z="PC3",
                color="Clase",
                symbol="Clase",
                title="Visualización 3D mediante PCA"
            )
            st.plotly_chart(fig_pca, use_container_width=True)

            if pca_variance:
                var = np.array(pca_variance)
                st.write(
                    f"Varianza explicada: PC1 = {var[0]:.4f}, "
                    f"PC2 = {var[1]:.4f}, PC3 = {var[2]:.4f}. "
                    f"Acumulada = {var.sum():.4f}."
                )

            st.caption(
                "El PCA se utiliza únicamente como recurso de "
                "visualización y no interviene en la selección del modelo."
            )
        else:
            st.info(
                "La información PCA aún no fue exportada desde el notebook."
            )

# ============================================================
# PREDICCIÓN INDIVIDUAL
# ============================================================
else:

    st.header("Predicción individual")

    st.write(
        "Seleccione un modelo y diligencie las características del empleado. "
        "La aplicación utilizará el modelo optimizado correspondiente."
    )

    st.info(
        "Esta predicción es una estimación del modelo. No representa "
        "una determinación definitiva sobre el comportamiento futuro "
        "de un empleado."
    )

    with st.form("formulario_prediccion"):

        datos = {}

        if categoricas:
            st.subheader("Información categórica")
            cols = st.columns(2)

            for i, variable in enumerate(categoricas):
                opciones = categoricas[variable]

                with cols[i % 2]:
                    datos[variable] = st.selectbox(
                        variable.replace("_", " "),
                        opciones
                    )

        numericas = [
            c for c in defaults
            if c not in categoricas
        ]

        if numericas:
            st.subheader("Información numérica")
            cols = st.columns(2)

            for i, variable in enumerate(numericas):

                minimo = float(mins[variable])
                maximo = float(maxs[variable])
                valor = float(defaults[variable])

                with cols[i % 2]:

                    if minimo == maximo:
                        datos[variable] = st.number_input(
                            variable.replace("_", " "),
                            value=valor,
                            disabled=True
                        )
                    else:
                        valor = min(
                            max(valor, minimo),
                            maximo
                        )

                        datos[variable] = st.number_input(
                            variable.replace("_", " "),
                            min_value=minimo,
                            max_value=maximo,
                            value=valor,
                            step=1.0
                        )

        ejecutar = st.form_submit_button(
            "Realizar predicción",
            type="primary"
        )

    if ejecutar:

        entrada = pd.DataFrame([datos])

        entrada_encoded = pd.get_dummies(
            entrada,
            drop_first=True,
            dtype=int
        )

        entrada_encoded = entrada_encoded.reindex(
            columns=columnas_modelo,
            fill_value=0
        )

        modelo = modelos[modelo_seleccionado]

        prediccion = int(
            modelo.predict(entrada_encoded)[0]
        )

        probabilidad = float(
            modelo.predict_proba(
                entrada_encoded
            )[0, 1]
        )

        st.divider()
        st.subheader("Resultado")

        if prediccion == 1:
            st.error("Resultado: posible abandono")
        else:
            st.success("Resultado: no abandono")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Probabilidad de abandono",
                f"{probabilidad:.2%}"
            )

        with c2:
            st.metric(
                "Probabilidad de permanencia",
                f"{1 - probabilidad:.2%}"
            )

        fig_prob = px.bar(
            x=["Abandono", "Permanencia"],
            y=[probabilidad, 1 - probabilidad],
            labels={
                "x": "",
                "y": "Probabilidad"
            },
            title="Probabilidades estimadas"
        )
        fig_prob.update_yaxes(
            range=[0, 1],
            tickformat=".0%"
        )

        st.plotly_chart(
            fig_prob,
            use_container_width=True
        )

st.divider()
st.caption(
    "Proyecto académico – Especialización en Inteligencia Artificial. "
    "Los modelos corresponden a la ejecución documentada en el notebook ACA 2."
)
