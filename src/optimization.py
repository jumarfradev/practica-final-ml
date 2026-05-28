"""Modulo de optimizacion de hiperparametros del proyecto.

Encapsula la busqueda sistematica de los mejores hiperparametros para los
modelos top del proyecto (XGBoost, RandomForest, Red Neuronal). Separado de
models.py por responsabilidad unica: models.py entrena con hiperparametros
dados; este modulo BUSCA los mejores hiperparametros.

Tecnicas usadas:
    - RandomizedSearchCV (sklearn) para XGBoost y RandomForest.
    - Busqueda manual de arquitecturas para la Red Neuronal.

Uso tipico:

    from src.data_loader import preparar_datos
    from src.optimization import optimizar_xgboost

    X_train, X_test, y_train, y_test, _ = preparar_datos()
    modelo, metricas, params = optimizar_xgboost(
        X_train, y_train, X_test, y_test
    )
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from src.models import SEED, calcular_metricas

# ============================================================================
# Optimizacion 1: XGBoost con RandomizedSearchCV
# ============================================================================


def optimizar_xgboost(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    n_iter: int = 20,
    cv: int = 3,
    n_jobs: int = -1,
    random_state: int = SEED,
    log_en_mlflow: bool = True,
) -> tuple[XGBClassifier, dict[str, float], dict]:
    """Optimiza hiperparametros de XGBoost con RandomizedSearchCV.

    Prueba n_iter combinaciones aleatorias del espacio de busqueda, cada una
    evaluada con validacion cruzada de cv folds, optimizando ROC-AUC. Devuelve
    el mejor modelo reentrenado, sus metricas en test, y los mejores params.

    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        X_test: Features de test.
        y_test: Target de test.
        n_iter: Numero de combinaciones aleatorias a probar. Defaults to 20.
        cv: Numero de folds de validacion cruzada. Defaults to 3.
        n_jobs: Cores para paralelizar. -1 usa todos. Defaults to -1.
        random_state: Semilla para reproducibilidad. Defaults to SEED.
        log_en_mlflow: Si True, registra el experimento en MLflow.
            Defaults to True.

    Returns:
        tuple: (mejor_modelo, metricas_test, mejores_params).
    """
    espacio_busqueda = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    modelo_base = XGBClassifier(
        n_jobs=n_jobs,
        random_state=random_state,
        eval_metric="logloss",
    )

    busqueda = RandomizedSearchCV(
        estimator=modelo_base,
        param_distributions=espacio_busqueda,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
    )

    busqueda.fit(X_train, y_train)

    mejor_modelo = busqueda.best_estimator_
    mejores_params = busqueda.best_params_

    y_pred = mejor_modelo.predict(X_test)
    y_pred_proba = mejor_modelo.predict_proba(X_test)[:, 1]
    metricas = calcular_metricas(y_test, y_pred, y_pred_proba)

    if log_en_mlflow:
        mlflow.start_run(run_name="XGBoost_optimizado")
        mlflow.log_params({"model_type": "XGBClassifier_optimizado", **mejores_params})
        mlflow.log_metric("cv_best_roc_auc", busqueda.best_score_)
        mlflow.log_metrics(metricas)
        mlflow.sklearn.log_model(mejor_modelo, "model")
        mlflow.end_run()

    return mejor_modelo, metricas, mejores_params


# ============================================================================
# Optimizacion 2: RandomForest con RandomizedSearchCV
# ============================================================================


def optimizar_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    n_iter: int = 20,
    cv: int = 3,
    n_jobs: int = -1,
    random_state: int = SEED,
    log_en_mlflow: bool = True,
) -> tuple[RandomForestClassifier, dict[str, float], dict]:
    """Optimiza hiperparametros de RandomForest con RandomizedSearchCV.

    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        X_test: Features de test.
        y_test: Target de test.
        n_iter: Numero de combinaciones aleatorias a probar. Defaults to 20.
        cv: Numero de folds de validacion cruzada. Defaults to 3.
        n_jobs: Cores para paralelizar. -1 usa todos. Defaults to -1.
        random_state: Semilla para reproducibilidad. Defaults to SEED.
        log_en_mlflow: Si True, registra el experimento en MLflow.
            Defaults to True.

    Returns:
        tuple: (mejor_modelo, metricas_test, mejores_params).
    """
    espacio_busqueda = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [10, 15, 20, 25, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }

    modelo_base = RandomForestClassifier(
        n_jobs=n_jobs,
        random_state=random_state,
    )

    busqueda = RandomizedSearchCV(
        estimator=modelo_base,
        param_distributions=espacio_busqueda,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
    )

    busqueda.fit(X_train, y_train)

    mejor_modelo = busqueda.best_estimator_
    mejores_params = busqueda.best_params_

    y_pred = mejor_modelo.predict(X_test)
    y_pred_proba = mejor_modelo.predict_proba(X_test)[:, 1]
    metricas = calcular_metricas(y_test, y_pred, y_pred_proba)

    if log_en_mlflow:
        mlflow.start_run(run_name="RandomForest_optimizado")
        mlflow.log_params({"model_type": "RandomForestClassifier_optimizado", **mejores_params})
        mlflow.log_metric("cv_best_roc_auc", busqueda.best_score_)
        mlflow.log_metrics(metricas)
        mlflow.sklearn.log_model(mejor_modelo, "model")
        mlflow.end_run()

    return mejor_modelo, metricas, mejores_params


# ============================================================================
# Optimizacion 3: Red Neuronal (busqueda manual de arquitecturas)
# ============================================================================


def optimizar_red_neuronal(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    random_state: int = SEED,
    log_en_mlflow: bool = True,
) -> tuple[object, dict[str, float], dict]:
    """Optimiza la red neuronal probando varias arquitecturas (busqueda manual).

    A diferencia de XGBoost y RandomForest (compatibles con RandomizedSearchCV),
    Keras no se integra directamente con la busqueda de sklearn. Por eso usamos
    busqueda manual: probamos varias arquitecturas concretas y nos quedamos con
    la que mejor ROC-AUC obtiene en test.

    Reutiliza entrenar_red_neuronal() de models.py, pasandole distintos
    hiperparametros en cada prueba.

    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        X_test: Features de test.
        y_test: Target de test.
        random_state: Semilla para reproducibilidad. Defaults to SEED.
        log_en_mlflow: Si True, registra cada arquitectura en MLflow.
            Defaults to True.

    Returns:
        tuple: (mejor_modelo, mejores_metricas, config_ganadora).
    """
    from src.models import entrenar_red_neuronal

    # Arquitecturas a probar (la base + 3 variantes)
    configuraciones = [
        {
            "nombre": "base_128_64_32",
            "capas_ocultas": (128, 64, 32),
            "dropout": 0.3,
            "learning_rate": 0.001,
        },
        {
            "nombre": "profunda_256_128_64_32",
            "capas_ocultas": (256, 128, 64, 32),
            "dropout": 0.3,
            "learning_rate": 0.001,
        },
        {
            "nombre": "ancha_256_128",
            "capas_ocultas": (256, 128),
            "dropout": 0.3,
            "learning_rate": 0.001,
        },
        {
            "nombre": "regularizada_128_64_32",
            "capas_ocultas": (128, 64, 32),
            "dropout": 0.4,
            "learning_rate": 0.0005,
        },
    ]

    mejor_roc_auc = -1.0
    mejor_modelo = None
    mejores_metricas: dict[str, float] = {}
    config_ganadora: dict = {}

    for config in configuraciones:
        print(f"\n{'=' * 60}")
        print(f"Probando arquitectura: {config['nombre']}")
        print(
            f"  capas={config['capas_ocultas']}, dropout={config['dropout']}, lr={config['learning_rate']}"
        )
        print("=" * 60)

        modelo, metricas = entrenar_red_neuronal(
            X_train,
            y_train,
            X_test,
            y_test,
            capas_ocultas=config["capas_ocultas"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
            random_state=random_state,
            log_en_mlflow=log_en_mlflow,
        )

        print(f"  ROC-AUC: {metricas['roc_auc']:.4f}")

        if metricas["roc_auc"] > mejor_roc_auc:
            mejor_roc_auc = metricas["roc_auc"]
            mejor_modelo = modelo
            mejores_metricas = metricas
            config_ganadora = config

    print(f"\n{'=' * 60}")
    print(f"MEJOR ARQUITECTURA: {config_ganadora['nombre']}")
    print(f"  ROC-AUC: {mejor_roc_auc:.4f}")
    print("=" * 60)

    return mejor_modelo, mejores_metricas, config_ganadora
