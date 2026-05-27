"""Módulo de modelado del proyecto.

Encapsula el entrenamiento de los 5 modelos obligatorios y su integración
con MLflow para tracking de experimentos.

Modelos implementados:
    - LogisticRegression (baseline)
    - DecisionTreeClassifier
    - RandomForestClassifier
    - XGBClassifier (Gradient Boosting)
    - Red Neuronal con Keras

Uso típico:

    from src.data_loader import preparar_datos
    from src.models import entrenar_modelo_logistic, evaluar_modelo

    X_train, X_test, y_train, y_test, _ = preparar_datos()
    modelo, metricas = entrenar_modelo_logistic(X_train, y_train, X_test, y_test)
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ============================================================================
# Constantes del módulo
# ============================================================================

PATH_PROYECTO: Path = Path(__file__).resolve().parent.parent
PATH_MLRUNS: Path = PATH_PROYECTO / "mlruns"

EXPERIMENT_NAME: str = "cancelaciones_hoteleras"
"""Nombre del experimento en MLflow donde se agruparán los runs."""

SEED: int = 42


# ============================================================================
# Setup de MLflow
# ============================================================================


def configurar_mlflow(
    nombre_experimento: str = EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> None:
    """Configura MLflow para que registre los experimentos localmente.

    Crea el experimento si no existe. Por defecto guarda los runs en la
    carpeta mlruns/ del proyecto.

    Args:
        nombre_experimento (str, optional): Nombre del experimento.
            Defaults to EXPERIMENT_NAME.
        tracking_uri (str | None, optional): URI de tracking. Si es None,
            usa file:///mlruns/ en la raíz del proyecto. Defaults to None.

    Example:
        >>> configurar_mlflow()
    """
    if tracking_uri is None:
        tracking_uri = f"file:///{PATH_MLRUNS.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(nombre_experimento)
    print("MLflow configurado.")
    print(f"  Tracking URI: {tracking_uri}")
    print(f"  Experimento: {nombre_experimento}")


# ============================================================================
# Funciones de evaluación
# ============================================================================


def calcular_metricas(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Calcula las métricas estándar de clasificación binaria.

    Args:
        y_true: Etiquetas reales (0/1).
        y_pred: Predicciones del modelo (0/1).
        y_pred_proba: Probabilidades predichas para la clase positiva.
            Necesarias para ROC-AUC. Si None, ROC-AUC no se calcula.

    Returns:
        dict[str, float]: Diccionario con accuracy, precision, recall, f1,
            y roc_auc (si y_pred_proba está disponible).
    """
    metricas = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    if y_pred_proba is not None:
        metricas["roc_auc"] = roc_auc_score(y_true, y_pred_proba)

    return metricas


def imprimir_metricas(metricas: dict[str, float], nombre_modelo: str = "") -> None:
    """Imprime las métricas en un formato legible.

    Args:
        metricas: Diccionario devuelto por calcular_metricas().
        nombre_modelo: Nombre opcional del modelo para la cabecera.
    """
    print(f"\n{'=' * 60}")
    print(f"METRICAS: {nombre_modelo}")
    print("=" * 60)
    for nombre, valor in metricas.items():
        print(f"  {nombre:12s}: {valor:.4f}")


# ============================================================================
# Modelo 1: LogisticRegression
# ============================================================================


def entrenar_logistic_regression(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    max_iter: int = 1000,
    C: float = 1.0,
    class_weight: str | None = None,
    random_state: int = SEED,
    log_en_mlflow: bool = True,
) -> tuple[LogisticRegression, dict[str, float]]:
    """Entrena un modelo de regresion logistica y registra en MLflow.

    La regresion logistica es nuestro modelo baseline: simple, rapido,
    interpretable. Capta solo relaciones lineales entre features y target.

    Args:
        X_train: Features de entrenamiento (ya preprocesadas).
        y_train: Target de entrenamiento.
        X_test: Features de test.
        y_test: Target de test.
        max_iter: Iteraciones maximas del optimizador. Defaults to 1000.
        C: Inverso de la fuerza de regularizacion. Valores mas pequenos =
            mas regularizacion. Defaults to 1.0.
        class_weight: Si 'balanced', penaliza mas los errores en la clase
            minoritaria. None = sin balanceo. Defaults to None.
        random_state: Semilla para reproducibilidad. Defaults to SEED.
        log_en_mlflow: Si True, registra el experimento en MLflow.
            Defaults to True.

    Returns:
        tuple: (modelo_entrenado, dict_de_metricas).

    Example:
        >>> modelo, metricas = entrenar_logistic_regression(
        ...     X_train, y_train, X_test, y_test
        ... )
        >>> metricas['roc_auc']
        0.84xx
    """
    nombre_modelo = "LogisticRegression"
    if class_weight == "balanced":
        nombre_modelo += "_balanced"

    if log_en_mlflow:
        mlflow.start_run(run_name=nombre_modelo)
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "max_iter": max_iter,
                "C": C,
                "class_weight": str(class_weight),
                "random_state": random_state,
            }
        )

    modelo = LogisticRegression(
        max_iter=max_iter,
        C=C,
        class_weight=class_weight,
        random_state=random_state,
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]

    metricas = calcular_metricas(y_test, y_pred, y_pred_proba)

    if log_en_mlflow:
        mlflow.log_metrics(metricas)
        mlflow.sklearn.log_model(modelo, "model")
        mlflow.end_run()

    return modelo, metricas
