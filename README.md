# Práctica Final · Machine Learning y Deep Learning

> Sistema automático de clasificación binaria sobre cancelaciones de reservas hoteleras.
> Máster en IA, Cloud Computing y DevOps · PontIA.tech · 2026

---

## Autores

- **Juan Martínez Fraile** — Linkedin: https://www.linkedin.com/in/juan-martinez-fraile/
- **[Nombre de tu pareja]** — [email pareja]

> _Roles y reparto de trabajo detallados en la sección [Roles del equipo](#-roles-del-equipo)._

---

## Descripción del problema

Este proyecto resuelve un problema de **clasificación binaria** en el sector hotelero:
predecir si una reserva será cancelada (`is_canceled = 1`) o se completará (`is_canceled = 0`)
a partir de información del cliente, características de la reserva y comportamiento histórico.

### El dataset

- **Fuente:** dataset `hotel_bookings` proporcionado por el módulo.
- **Volumen:** 119.390 reservas, 32 variables.
- **Variable objetivo:** `is_canceled` (binaria, ~37% cancelaciones).
- **Tipos de variable:** mezcla de numéricas continuas, enteras y categóricas (hotel, mes, país, canal, tipo de cliente, etc.).

### Por qué este problema importa

Las cancelaciones de reservas son uno de los principales problemas operativos
de la industria hotelera: afectan a la planificación de personal, al precio dinámico,
y al overbooking. Un modelo que anticipe la probabilidad de cancelación
permite ajustar precios, ofrecer incentivos al cliente para mantener la reserva,
o reasignar habitaciones a tiempo.

> _Detalle completo del problema, análisis exploratorio y diseño en `docs/informe_final.md`._

---

## Estructura del proyecto

```
practica-final-ml/
├── .gitignore
├── README.md                         # Este archivo
├── requirements.txt                  # Dependencias del proyecto
├── trainer.py                        # Script principal: orquesta todo el pipeline
│
├── data/
│   ├── raw/                          # Dataset original (inmutable)
│   └── processed/                    # Datos preprocesados (regenerable)
│
├── docs/
│   └── informe_final.md              # Informe completo de la práctica
│
├── models/
│   ├── tests/                        # Modelos intermedios de pruebas
│   └── best_model.pkl                # Mejor modelo seleccionado
│
├── notebooks/
│   ├── exploracion/                  # Notebooks iterativos (sucios)
│   │   ├── eda_inicial.ipynb
│   │   └── pruebas_modelos.ipynb
│   └── finales/                      # Versiones limpias para presentar
│       ├── eda_final.ipynb
│       └── comparativa_modelos.ipynb
│
├── outputs/                          # Gráficas y resultados generados
│
└── src/                              # Código fuente del pipeline
    ├── __init__.py
    ├── config.py                     # Constantes y rutas
    ├── data_loader.py                # Carga y preprocesamiento
    ├── model_trainer.py              # Entrenamiento de modelos
    ├── evaluator.py                  # Métricas y visualizaciones
    └── predictor.py                  # Inferencia con modelos entrenados
```

---

## Requisitos e instalación

### Requisitos previos

- **Python 3.11** (gestionado automáticamente por `uv`)
- **[`uv`](https://docs.astral.sh/uv/)** — gestor de entornos y dependencias
- **git**

### Instalación paso a paso

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/practica-final-ml.git
cd practica-final-ml

# 2. Crear el entorno virtual con uv (Python 3.11)
uv venv --python 3.11 env-pontia-ml

# 3. Activar el entorno
# Windows (Git Bash):
source env-pontia-ml/Scripts/activate
# Mac / Linux:
source env-pontia-ml/bin/activate

# 4. Instalar dependencias
uv pip install -r requirements.txt
```

### Verificar la instalación

```bash
python --version       # Debe mostrar: Python 3.11.x
python -c "import pandas, sklearn, tensorflow, xgboost; print('OK')"
```

---

## Cómo ejecutar el proyecto

> _TODO: completar cuando el pipeline esté implementado_

### Pipeline completo

```bash
# Entrenar y evaluar todos los modelos, guardar el mejor
python trainer.py
```

### Predicciones con el modelo entrenado

```bash
python -m src.predictor --input ruta/al/csv_nuevo.csv
```

### Exploración con notebooks

Los notebooks de `notebooks/exploracion/` documentan el proceso de análisis.
Los de `notebooks/finales/` son las versiones presentables para la defensa.

---

## Roles del equipo

> _TODO: completar tras acordar el reparto con la pareja_

| Integrante | Responsabilidad principal | Aportaciones concretas |
|---|---|---|
| **[Tu nombre]** | _por definir_ | _por definir_ |
| **[Pareja]** | _por definir_ | _por definir_ |

---

## Resultados y conclusiones

> _TODO: completar al finalizar la fase de modelado_

### Métrica principal seleccionada

> _TODO: justificar la elección (accuracy / F1 / AUC-ROC / ...) según el problema de negocio_

### Comparativa de modelos

> _TODO: tabla con accuracy, F1-score, ROC-AUC de los 5+ modelos entrenados_

### Modelo final elegido

> _TODO: nombre del modelo, sus hiperparámetros y razonamiento de la elección_

---

## Documentación adicional

- **Informe final completo:** [`docs/informe_final.md`](docs/informe_final.md)
- **Análisis exploratorio:** [`notebooks/finales/eda_final.ipynb`](notebooks/finales/eda_final.ipynb)
- **Comparativa de modelos:** [`notebooks/finales/comparativa_modelos.ipynb`](notebooks/finales/comparativa_modelos.ipynb)

---

## Tecnologías utilizadas

- **Datos y análisis:** pandas, numpy, matplotlib, seaborn, plotly
- **Modelos clásicos:** scikit-learn, xgboost, lightgbm
- **Deep Learning:** tensorflow, keras
- **Gestión de experimentos:** mlflow _(bonus)_
- **Entorno y dependencias:** uv, Python 3.11
- **Calidad de código y DevOps** Black + Ruff (formateo y linting), pre-commit (hooks locales), GitHub Actions (CI: verificación automática en cada push)

---

## Desarrollo

Si vas a contribuir al código, las herramientas de calidad están configuradas
automáticamente. Aquí algunos comandos útiles:

### Verificar formato y linting

```bash
# Linter (detecta problemas)
ruff check .

# Formateador (cambia archivos)
black .
ruff format .

# Ejecutar todos los hooks de pre-commit manualmente
pre-commit run --all-files
```

### Verificar que las dependencias se instalan limpias

```bash
python -c "import pandas, sklearn, tensorflow, xgboost, lightgbm, keras, mlflow; print('OK')"
```

### Pre-commit hooks

Tras clonar el repo, ejecuta una vez:

```bash
pre-commit install
```

A partir de ahí, cada `git commit` ejecutará automáticamente los checks de calidad.

---

## Cómo contribuir

Si trabajas en este proyecto, lee primero la [guía de contribución](CONTRIBUTING.md).
Define el flujo de ramas, convenciones de commits y cómo abrir Pull Requests.

---
