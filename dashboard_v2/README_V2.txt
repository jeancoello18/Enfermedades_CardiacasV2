# Dashboard CRISP-ML(Q) - Versión 2

## Cambio incorporado
La pestaña **Comparativa de modelos** ahora incluye los cuatro algoritmos:
- Random Forest 1
- Random Forest 2
- Regresión logística
- Red neuronal (MLP), correspondiente al modelo de Deep Learning

La comparación utiliza las métricas exportadas por el notebook en `comparativa_examen.csv`.
También muestra la gráfica comparativa, las curvas ROC y las matrices de confusión generadas por el notebook.
La pestaña de predicción conserva el funcionamiento de la versión 1.

## Archivos
Mantén todos los archivos de esta carpeta juntos. No cambies los nombres de los archivos de datos ni de las imágenes.

## Ejecución local
1. Instala Python.
2. Abre una terminal dentro de esta carpeta.
3. Ejecuta: `pip install -r requirements.txt`
4. Ejecuta: `streamlit run app_comparativa.py`

El navegador abrirá el dashboard localmente.

## Nota metodológica
La red neuronal se incorpora a la **comparación de resultados** mediante las métricas y visualizaciones exportadas desde el notebook. El archivo `archivos_dashboard (1).zip` no contiene `pesos_mlp.npz`, por lo que esta versión no usa la MLP para generar predicciones individuales en la pestaña Predicción. Esa pestaña conserva los tres modelos de la versión 1.
