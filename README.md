# Clasificación del abandono de empleados

Aplicación académica de la Especialización en Inteligencia Artificial.

## Contenido
- `app.py`: aplicación Streamlit independiente para clasificación del abandono, con portada y sección EDA.
- `WA_Fn-UseC_-HR-Employee-Attrition.csv`: dataset usado por el EDA.
- `requirements.txt`: dependencias.

## Importante: archivo de modelos requerido
La aplicación original carga `modelos_despliegue.joblib` para conservar los modelos entrenados, métricas, matrices de confusión y resultados exportados. Ese archivo no estaba incluido en el `app(6).py` adjunto ni en los archivos disponibles para esta edición. **Debe copiarse a esta carpeta desde el despliegue original antes de ejecutar la aplicación**, sin regenerar ni cambiar los modelos.

## Ejecución
1. Instalar dependencias: `pip install -r requirements.txt`
2. Colocar `modelos_despliegue.joblib` en la misma carpeta que `app.py`.
3. Ejecutar: `streamlit run app.py`

La aplicación conserva los módulos de clasificación existentes y agrega:
- Portada con los integrantes y el docente Hamilton Rivera.
- EDA con calidad de datos, distribuciones, relaciones, valores atípicos y hallazgos.

No incluye regresión de ingresos ni segmentación, porque esta versión es exclusivamente de clasificación.
