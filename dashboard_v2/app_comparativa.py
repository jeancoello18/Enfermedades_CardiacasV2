"""
Dashboard: comparativa de modelos para predecir enfermedad cardíaca.
Trabajo autónomo 1-2 - Grupo 4 - Minería de Datos (CRISP-ML, Fase 5: Despliegue)

Ejecución:  python -m streamlit run app_comparativa.py
Requiere el archivo processed.cleveland.data en la misma carpeta.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed.cleveland.data")
COLUMNAS = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholestoral',
            'fasting_blood_sugar', 'electrocardiographic', 'max_heart_rate',
            'induced_angina', 'ST_depression', 'slope', 'no_of_vessels', 'thal', 'diagnosis']
MEJOR_MODELO = "Random Forest 2"

st.set_page_config(page_title="Comparativa de modelos - Enfermedad cardíaca", page_icon="❤️", layout="wide")


# ---------------------------------------------------------------------------
# Datos y modelos (mismo procedimiento que el notebook)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv(RUTA, names=COLUMNAS, header=None, na_values="?")
    df['no_of_vessels'] = df['no_of_vessels'].fillna(df['no_of_vessels'].mode()[0])
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x > 0 else 0)
    return df


def metricas(modelo, X_, y_):
    pred = modelo.predict(X_)
    prob = modelo.predict_proba(X_)[:, 1]
    return {"Exactitud": accuracy_score(y_, pred), "Precisión": precision_score(y_, pred),
            "Recall": recall_score(y_, pred), "F1": f1_score(y_, pred),
            "AUC": roc_auc_score(y_, prob)}


@st.cache_resource
def entrenar_modelos():
    df = cargar_datos()
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2606)

    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)

    modelos = {
        # Modelo de la práctica 1-2: hiperparámetros por defecto
        "Random Forest 1": RandomForestClassifier(n_estimators=100, random_state=2606),
        # Hiperparámetros elegidos con GridSearchCV (validación cruzada) en el notebook
        "Random Forest 2": RandomForestClassifier(n_estimators=100, max_depth=3,
                                                  min_samples_leaf=3, random_state=2606),
        # Algoritmo nuevo
        "Regresión logística": LogisticRegression(max_iter=1000, random_state=2606),
    }
    for m in modelos.values():
        m.fit(X_train_esc, y_train)

    tabla_train = pd.DataFrame({n: metricas(m, X_train_esc, y_train) for n, m in modelos.items()}).T.round(3)
    tabla_test = pd.DataFrame({n: metricas(m, X_test_esc, y_test) for n, m in modelos.items()}).T.round(3)
    datos = {"train": (X_train_esc, y_train), "test": (X_test_esc, y_test)}
    return modelos, escalador, tabla_train, tabla_test, datos


def probabilidad(modelo, escalador, valores):
    fila = pd.DataFrame([valores], columns=COLUMNAS[:-1])
    return modelo.predict_proba(escalador.transform(fila))[0, 1]


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
if not os.path.exists(RUTA):
    st.error("No se encontró 'processed.cleveland.data'. Colócalo en la misma carpeta que este archivo.")
    st.stop()

df = cargar_datos()
modelos, escalador, tabla_train, tabla_test, datos = entrenar_modelos()

st.title("❤️ Predicción de enfermedad cardíaca: comparativa de modelos")
st.caption("Caso de estudio: Mishra et al. (2024) · Dataset Cleveland (UCI) · CRISP-ML, Fase 5: Despliegue")

with st.sidebar:
    st.header("Modelo para predecir")
    nombres = list(modelos.keys())
    nombre_modelo = st.selectbox("Modelo", options=nombres, index=nombres.index(MEJOR_MODELO))
    st.write(f"**Exactitud en prueba:** {tabla_test.loc[nombre_modelo, 'Exactitud']:.2%}")
    st.write(f"**Recall en prueba:** {tabla_test.loc[nombre_modelo, 'Recall']:.3f}")
    st.write(f"**AUC en prueba:** {tabla_test.loc[nombre_modelo, 'AUC']:.3f}")
    st.caption(f"Por defecto se usa {MEJOR_MODELO}, el de mejor desempeño en prueba.")
    st.warning("Uso académico. No sustituye un diagnóstico médico.")

tab_pred, tab_comp, tab_datos = st.tabs(["Predicción", "Comparativa de modelos", "Datos"])

# ---- Pestaña 1: predicción ----
with tab_pred:
    st.subheader("Datos clínicos del paciente")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Edad (años)", min_value=20, max_value=90, value=55)
        sex = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: {0: "Mujer", 1: "Hombre"}[x])
        chest_pain = st.selectbox(
            "Tipo de dolor de pecho", options=[1, 2, 3, 4],
            format_func=lambda x: {1: "1 - Angina típica", 2: "2 - Angina atípica",
                                   3: "3 - Dolor no anginoso", 4: "4 - Asintomático"}[x])
        blood_pressure = st.number_input("Presión arterial en reposo (mm Hg)", min_value=80, max_value=220, value=130)
        serum_cholestoral = st.number_input("Colesterol (mg/dl)", min_value=100, max_value=600, value=245)
    with c2:
        fasting_blood_sugar = st.selectbox(
            "Azúcar en ayunas > 120 mg/dl", options=[0, 1], format_func=lambda x: {0: "No", 1: "Sí"}[x])
        electrocardiographic = st.selectbox(
            "Electrocardiograma en reposo", options=[0, 1, 2],
            format_func=lambda x: {0: "0 - Normal", 1: "1 - Anomalía ST-T", 2: "2 - Hipertrofia"}[x])
        max_heart_rate = st.number_input("Frecuencia cardíaca máxima", min_value=60, max_value=220, value=150)
        induced_angina = st.selectbox(
            "Angina inducida por ejercicio", options=[0, 1], format_func=lambda x: {0: "No", 1: "Sí"}[x])
    with c3:
        ST_depression = st.number_input("Depresión ST por ejercicio", min_value=0.0, max_value=7.0,
                                        value=1.0, step=0.1)
        slope = st.selectbox(
            "Pendiente del segmento ST", options=[1, 2, 3],
            format_func=lambda x: {1: "1 - Ascendente", 2: "2 - Plana", 3: "3 - Descendente"}[x])
        no_of_vessels = st.selectbox("Vasos principales coloreados (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox(
            "Thal", options=[3, 6, 7],
            format_func=lambda x: {3: "3 - Normal", 6: "6 - Defecto fijo", 7: "7 - Defecto reversible"}[x])

    if st.button("Predecir"):
        valores = [age, sex, chest_pain, blood_pressure, serum_cholestoral, fasting_blood_sugar,
                   electrocardiographic, max_heart_rate, induced_angina, ST_depression, slope,
                   no_of_vessels, thal]
        prob = probabilidad(modelos[nombre_modelo], escalador, valores)
        st.metric(f"Probabilidad de enfermedad cardíaca ({nombre_modelo})", f"{prob:.1%}")
        st.progress(float(prob))
        if prob >= 0.5:
            st.error("Predicción: **ENFERMO (1)**. Se recomienda evaluación médica.")
        else:
            st.success("Predicción: **SANO (0)**.")
        st.write("Probabilidad de enfermedad según cada modelo:")
        comparacion = pd.DataFrame(
            {"Probabilidad": {n: f"{probabilidad(m, escalador, valores):.1%}" for n, m in modelos.items()}})
        st.dataframe(comparacion)

# ---- Pestaña 2: comparativa de los cuatro modelos ----
with tab_comp:
    st.subheader("Métricas de los cuatro modelos")
    ruta_comparativa = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparativa_examen.csv")
    if not os.path.exists(ruta_comparativa):
        st.error("No se encontró comparativa_examen.csv. Colócalo junto a app_comparativa.py.")
    else:
        comparativa_df = pd.read_csv(ruta_comparativa)
        orden_modelos = ["Random Forest 1", "Random Forest 2",
                         "Regresión logística", "Red neuronal (MLP)"]
        metricas_visibles = ["Exactitud", "Precisión", "Recall",
                             "Especificidad", "F1", "AUC"]

        col_train, col_test = st.columns(2)
        with col_train:
            st.write("**Entrenamiento (212 pacientes)**")
            tabla_train_v2 = comparativa_df[comparativa_df["Conjunto"] == "Entrenamiento"].set_index("Algoritmo")
            st.dataframe(tabla_train_v2.reindex(orden_modelos)[["Tipo"] + metricas_visibles],
                         use_container_width=True)
        with col_test:
            st.write("**Prueba (91 pacientes)**")
            tabla_test_v2 = comparativa_df[comparativa_df["Conjunto"] == "Prueba"].set_index("Algoritmo")
            st.dataframe(tabla_test_v2.reindex(orden_modelos)[["Tipo"] + metricas_visibles],
                         use_container_width=True)

        st.subheader("Comparación visual de métricas en prueba")
        grafica = tabla_test_v2.reindex(orden_modelos)[metricas_visibles]
        st.bar_chart(grafica, y=metricas_visibles, use_container_width=True)

        st.subheader("Curvas ROC")
        ruta_roc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roc_curvas.png")
        if os.path.exists(ruta_roc):
            st.image(ruta_roc, caption="Curvas ROC de los cuatro modelos (entrenamiento y prueba)",
                     use_container_width=True)
        else:
            st.warning("No se encontró roc_curvas.png.")

        st.subheader("Matrices de confusión (prueba)")
        ruta_matrices = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matrices_confusion.png")
        if os.path.exists(ruta_matrices):
            st.image(ruta_matrices, caption="Matrices de confusión de los cuatro modelos",
                     use_container_width=True)
        else:
            st.warning("No se encontró matrices_confusion.png.")

        with st.expander("Curvas de aprendizaje de la red neuronal (MLP)"):
            ruta_aprendizaje = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curvas_aprendizaje.png")
            if os.path.exists(ruta_aprendizaje):
                st.image(ruta_aprendizaje, use_container_width=True)
            else:
                st.info("No se encontró curvas_aprendizaje.png.")

        st.download_button(
            "Descargar métricas de los cuatro modelos (CSV)",
            data=comparativa_df.to_csv(index=False).encode("utf-8"),
            file_name="comparativa_modelos_v2.csv",
            mime="text/csv"
        )

# ---- Pestaña 3: datos ----
with tab_datos:
    st.subheader("Conjunto de datos preparado")
    st.write(f"{df.shape[0]} pacientes y {df.shape[1]} columnas (diagnosis: 0 = sano, 1 = enfermo).")
    st.dataframe(df.head(10))
