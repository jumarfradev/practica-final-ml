"""Script principal de entrenamiento del proyecto.

Orquesta el flujo completo de Machine Learning desde los datos crudos hasta
la persistencia del mejor modelo, cumpliendo el requisito del enunciado de
"automatizar el flujo completo desde los datos hasta la inferencia".

Flujo ejecutado:
    1. Preparar datos (carga, limpieza, preprocesamiento, split estratificado).
    2. Congelar las categorias top-N (country, agent) para inferencia futura.
    3. Entrenar los 5 modelos base.
    4. Entrenar las 3 optimizaciones (XGBoost, RandomForest, Red Neuronal).
    5. Persistir las metricas de los 8 modelos en JSON.
    6. Seleccionar el mejor modelo por ROC-AUC.
    7. Guardar el bundle de inferencia (best_model.pkl) con todo lo necesario
       para predecir sobre datos crudos: modelo + preprocesador + categorias
       top-N congeladas + nombres de features + metadatos.

Uso:
    python trainer.py                 # flujo completo (8 modelos)
    python trainer.py --rapido        # solo 5 modelos base (sin optimizacion)
    python trainer.py --sin-mlflow    # desactiva el registro en MLflow

El modelo ganador se guarda en models/best_model.pkl. Si el ganador fuese la
red neuronal (Keras), el .pkl guarda los metadatos y el modelo Keras se guarda
aparte en models/best_model.keras (Keras no serializa de forma fiable con
joblib/pickle).
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import joblib

from src.data_loader import (
    cargar_dataset,
    limpiar_dataset,
    obtener_categorias_top_n,
    preparar_datos,
)
from src.metrics_io import guardar_metricas
from src.models import (
    configurar_mlflow,
    entrenar_decision_tree,
    entrenar_logistic_regression,
    entrenar_random_forest,
    entrenar_red_neuronal,
    entrenar_xgboost,
)
from src.optimization import (
    optimizar_random_forest,
    optimizar_red_neuronal,
    optimizar_xgboost,
)

# ============================================================================
# Constantes
# ============================================================================

PATH_PROYECTO: Path = Path(__file__).resolve().parent
PATH_MODELS: Path = PATH_PROYECTO / "models"
PATH_BEST_MODEL: Path = PATH_MODELS / "best_model.pkl"
PATH_BEST_MODEL_KERAS: Path = PATH_MODELS / "best_model.keras"

METRICA_PRINCIPAL: str = "roc_auc"
"""Metrica usada para seleccionar el mejor modelo (ver justificacion en el
informe: el dataset esta desbalanceado 63/37 y ROC-AUC es robusta a ello)."""

# Modelos que NO son sklearn-compatibles para serializar con joblib.
NOMBRES_MODELOS_KERAS: set[str] = {"RedNeuronal", "RedNeuronal_OPT"}


# ============================================================================
# Entrenamiento de los modelos
# ============================================================================


def entrenar_todos_los_modelos(
    X_train,
    X_test,
    y_train,
    y_test,
    rapido: bool = False,
    log_en_mlflow: bool = True,
) -> dict[str, tuple]:
    """Entrena los modelos base (y opcionalmente las optimizaciones).

    Args:
        X_train, X_test, y_train, y_test: Conjuntos ya preprocesados.
        rapido (bool): Si True, entrena solo los 5 modelos base. Si False,
            anade las 3 optimizaciones con RandomizedSearchCV. Defaults to False.
        log_en_mlflow (bool): Si True, registra cada run en MLflow.

    Returns:
        dict[str, tuple]: Diccionario {nombre_modelo: (modelo, metricas)}.
    """
    resultados: dict[str, tuple] = {}

    print("\n" + "=" * 70)
    print("ENTRENANDO MODELOS BASE")
    print("=" * 70)

    print("\n[1/5] LogisticRegression...")
    modelo, met = entrenar_logistic_regression(
        X_train, y_train, X_test, y_test, log_en_mlflow=log_en_mlflow
    )
    resultados["LogisticRegression"] = (modelo, met)
    guardar_metricas("LogisticRegression", met)

    print("\n[2/5] DecisionTree...")
    modelo, met = entrenar_decision_tree(
        X_train, y_train, X_test, y_test, log_en_mlflow=log_en_mlflow
    )
    resultados["DecisionTree"] = (modelo, met)
    guardar_metricas("DecisionTree", met)

    print("\n[3/5] RandomForest...")
    modelo, met = entrenar_random_forest(
        X_train, y_train, X_test, y_test, log_en_mlflow=log_en_mlflow
    )
    resultados["RandomForest"] = (modelo, met)
    guardar_metricas("RandomForest", met)

    print("\n[4/5] XGBoost...")
    modelo, met = entrenar_xgboost(X_train, y_train, X_test, y_test, log_en_mlflow=log_en_mlflow)
    resultados["XGBoost"] = (modelo, met)
    guardar_metricas("XGBoost", met)

    print("\n[5/5] RedNeuronal...")
    modelo, met = entrenar_red_neuronal(
        X_train, y_train, X_test, y_test, log_en_mlflow=log_en_mlflow
    )
    resultados["RedNeuronal"] = (modelo, met)
    guardar_metricas("RedNeuronal", met)

    if rapido:
        print("\n[modo rapido] Se omiten las optimizaciones.")
        return resultados

    print("\n" + "=" * 70)
    print("ENTRENANDO OPTIMIZACIONES (RandomizedSearchCV + busqueda manual)")
    print("=" * 70)

    print("\n[1/3] XGBoost optimizado...")
    modelo, met, _params = optimizar_xgboost(
        X_train, y_train, X_test, y_test, log_en_mlflow=log_en_mlflow
    )
    resultados["XGBoost_OPT"] = (modelo, met)
    guardar_metricas("XGBoost_OPT", met)

    print("\n[2/3] RandomForest optimizado...")
    modelo, met, _params = optimizar_random_forest(
        X_train, y_train, X_test, y_test, log_en_mlflow=log_en_mlflow
    )
    resultados["RandomForest_OPT"] = (modelo, met)
    guardar_metricas("RandomForest_OPT", met)

    print("\n[3/3] RedNeuronal optimizada...")
    modelo, met, _config = optimizar_red_neuronal(
        X_train, y_train, X_test, y_test, log_en_mlflow=log_en_mlflow
    )
    resultados["RedNeuronal_OPT"] = (modelo, met)
    guardar_metricas("RedNeuronal_OPT", met)

    return resultados


# ============================================================================
# Seleccion del mejor modelo
# ============================================================================


def seleccionar_mejor_modelo(
    resultados: dict[str, tuple],
    metrica: str = METRICA_PRINCIPAL,
) -> tuple[str, object, dict[str, float]]:
    """Selecciona el modelo con mayor valor en la metrica principal.

    Args:
        resultados: Diccionario {nombre: (modelo, metricas)}.
        metrica: Clave de la metrica para comparar. Defaults to ROC-AUC.

    Returns:
        tuple: (nombre_ganador, modelo_ganador, metricas_ganador).
    """
    print("\n" + "=" * 70)
    print(f"COMPARATIVA DE MODELOS (ordenado por {metrica})")
    print("=" * 70)

    tabla = sorted(
        resultados.items(),
        key=lambda kv: kv[1][1].get(metrica, -1.0),
        reverse=True,
    )

    print(f"\n{'Modelo':<22}{'ROC-AUC':>10}{'F1':>10}{'Accuracy':>11}")
    print("-" * 53)
    for nombre, (_modelo, met) in tabla:
        print(
            f"{nombre:<22}"
            f"{met.get('roc_auc', 0):>10.4f}"
            f"{met.get('f1', 0):>10.4f}"
            f"{met.get('accuracy', 0):>11.4f}"
        )

    nombre_ganador, (modelo_ganador, met_ganador) = tabla[0]

    print("\n" + "=" * 70)
    print(f"MODELO GANADOR: {nombre_ganador}")
    print(f"  {metrica}: {met_ganador.get(metrica, 0):.4f}")
    print("=" * 70)

    return nombre_ganador, modelo_ganador, met_ganador


# ============================================================================
# Persistencia del bundle de inferencia
# ============================================================================


def guardar_bundle(
    nombre_ganador: str,
    modelo_ganador: object,
    metricas_ganador: dict[str, float],
    preprocesador: object,
    categorias_top_n: dict[str, list],
    nombres_features: list[str],
) -> None:
    """Guarda el bundle de inferencia con todo lo necesario para predecir.

    El bundle (best_model.pkl) contiene no solo el modelo, sino toda la cadena
    de transformacion necesaria para que la API/predictor reciba datos CRUDOS
    y prediga: preprocesador fiteado + categorias top-N congeladas + nombres
    de features + metadatos.

    Si el ganador es la red neuronal (Keras), el modelo se guarda aparte en
    formato .keras y el bundle pkl guarda solo la referencia y metadatos.

    Args:
        nombre_ganador: Nombre del modelo ganador.
        modelo_ganador: El modelo entrenado.
        metricas_ganador: Metricas del modelo ganador.
        preprocesador: ColumnTransformer fiteado en train.
        categorias_top_n: Categorias top-N congeladas de country y agent.
        nombres_features: Nombres de las columnas de salida del preprocesador.
    """
    PATH_MODELS.mkdir(parents=True, exist_ok=True)

    es_keras = nombre_ganador in NOMBRES_MODELOS_KERAS

    bundle = {
        "nombre_modelo": nombre_ganador,
        "metrica_principal": METRICA_PRINCIPAL,
        "metricas": metricas_ganador,
        "preprocesador": preprocesador,
        "categorias_top_n": categorias_top_n,
        "nombres_features": nombres_features,
        "es_keras": es_keras,
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
    }

    if es_keras:
        # Guardar el modelo Keras aparte (joblib no lo serializa bien)
        modelo_ganador.save(PATH_BEST_MODEL_KERAS)
        bundle["modelo"] = None
        bundle["ruta_keras"] = PATH_BEST_MODEL_KERAS.name
        print(f"\nModelo Keras guardado en: {PATH_BEST_MODEL_KERAS}")
    else:
        bundle["modelo"] = modelo_ganador
        bundle["ruta_keras"] = None

    joblib.dump(bundle, PATH_BEST_MODEL)
    print(f"Bundle de inferencia guardado en: {PATH_BEST_MODEL}")
    print(f"  Modelo: {nombre_ganador}")
    print(f"  ROC-AUC: {metricas_ganador.get('roc_auc', 0):.4f}")
    print(f"  Features: {len(nombres_features)}")


# ============================================================================
# Orquestacion principal
# ============================================================================


def main(rapido: bool = False, log_en_mlflow: bool = True) -> None:
    """Ejecuta el pipeline completo de entrenamiento y persistencia.

    Args:
        rapido: Si True, solo 5 modelos base. Defaults to False.
        log_en_mlflow: Si True, registra en MLflow. Defaults to True.
    """
    inicio = datetime.now()

    print("=" * 70)
    print("PIPELINE DE ENTRENAMIENTO - Cancelaciones hoteleras")
    print(f"Modo: {'rapido (5 modelos)' if rapido else 'completo (8 modelos)'}")
    print(f"MLflow: {'activado' if log_en_mlflow else 'desactivado'}")
    print("=" * 70)

    if log_en_mlflow:
        configurar_mlflow()

    # --- 1. Preparar datos (split + preprocesador fiteado) ---
    print("\nPreparando datos...")
    X_train, X_test, y_train, y_test, preprocesador = preparar_datos()
    nombres_features = list(preprocesador.get_feature_names_out())
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # --- 2. Congelar categorias top-N para inferencia ---
    # Se recalcula la cadena de limpieza para obtener el df limpio del que
    # extraer el top-N (mismo orden que preparar_datos por dentro).
    df_limpio = limpiar_dataset(cargar_dataset())
    categorias_top_n = obtener_categorias_top_n(df_limpio)
    print(
        f"  Top-N congelado: {len(categorias_top_n['country'])} paises, "
        f"{len(categorias_top_n['agent'])} agentes"
    )

    # --- 3-4. Entrenar todos los modelos ---
    resultados = entrenar_todos_los_modelos(
        X_train, X_test, y_train, y_test, rapido=rapido, log_en_mlflow=log_en_mlflow
    )

    # --- 6. Seleccionar el mejor ---
    nombre_ganador, modelo_ganador, metricas_ganador = seleccionar_mejor_modelo(resultados)

    # --- 7. Guardar bundle de inferencia ---
    guardar_bundle(
        nombre_ganador,
        modelo_ganador,
        metricas_ganador,
        preprocesador,
        categorias_top_n,
        nombres_features,
    )

    duracion = (datetime.now() - inicio).total_seconds()
    print(f"\nPipeline completado en {duracion:.1f} segundos.")


def _parse_args() -> argparse.Namespace:
    """Parsea los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena y compara modelos de clasificacion binaria, "
        "selecciona el mejor por ROC-AUC y lo persiste para inferencia."
    )
    parser.add_argument(
        "--rapido",
        action="store_true",
        help="Entrena solo los 5 modelos base (omite las optimizaciones).",
    )
    parser.add_argument(
        "--sin-mlflow",
        action="store_true",
        help="Desactiva el registro de experimentos en MLflow.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        main(rapido=args.rapido, log_en_mlflow=not args.sin_mlflow)
    except FileNotFoundError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        print(
            "Asegurate de que el dataset esta en data/raw/ con el nombre correcto.",
            file=sys.stderr,
        )
        sys.exit(1)
