# AGROSENSE-AI

Servicio de machine learning de AgroSense. Expone cuatro modelos por HTTP:

| Modelo | Ruta | Algoritmo |
|---|---|---|
| Riego | `POST /predict/irrigation` | Gradient Boosting (clasificación + regresión) |
| Pronóstico de humedad (6, 12 y 24 h) | `POST /predict/moisture-forecast` | Regresión cuantílica + calibración conforme |
| Fallas de sensores | `POST /detect/anomalies` | Isolation Forest + reglas físicas |
| Riesgo de enfermedades | `POST /predict/disease-risk` | Gradient Boosting |

`/predict/irrigation` usa el mismo contrato que ya consume el backend (`AiClientService`), así que no hay
que cambiar nada allá.

Documentación de la API en `/docs` y página de prueba en `/demo`.

## Datos de entrenamiento

Como todavía no hay histórico real, los datos se generan con un simulador de balance hídrico del suelo
(evapotranspiración de Hargreaves, FAO-56, con el coeficiente `kc` de cada cultivo). Cultivos incluidos:
café, plátano, papa, maíz, cacao y uno genérico.

## Resultados

Sobre datos de prueba que el modelo no vio en el entrenamiento (`agrosense_ai/artifacts/metrics.json`):

| Modelo | Métrica principal |
|---|---|
| Riego | 94,6 % de exactitud (la regla anterior: 82,0 %), ROC-AUC 0,988 |
| Pronóstico | Error medio a 24 h de 1,66 puntos (suponer que no cambia: 4,70); los intervalos cubren el 81 % |
| Fallas | Picos, sensores congelados y valores imposibles: 100 % detectados; 1,3 % de falsas alarmas |
| Enfermedades | ROC-AUC 0,955 |

## Ejecutar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m agrosense_ai
```

Variables de entorno: `PORT` (8000 por defecto), `AI_SERVICE_TOKEN` (si se define, las rutas `POST` piden
`Authorization: Bearer <token>`) y `LOG_LEVEL`.

Entrenar de nuevo: `python -m agrosense_ai.training` (`--quick` para una prueba rápida).

Pruebas y linter: `python -m pytest` y `ruff check .`

## Conexión con el backend

En el backend: `AI_SERVICE_URL` con la URL de este servicio y `AI_SERVICE_TOKEN` con el mismo token.
En el plan gratuito de Render conviene subir `AI_SERVICE_TIMEOUT_MS` a `60000`, porque el servicio se duerme.

## Despliegue

Render: **New → Blueprint** con este repositorio (`render.yaml`). También funciona con Docker:

```bash
docker build -t agrosense-ai .
docker run -p 8000:8000 agrosense-ai
```

## Estructura

```
agrosense_ai/
  api.py          rutas y manejo de errores
  schemas.py      validación de peticiones
  services/       lógica de cada modelo
  training.py     entrenamiento y evaluación
  simulation.py   generación de datos
  crops.py        perfiles de cultivo
  artifacts/      modelos entrenados
tests/
```
