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
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBClassifier

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


# ============================================================================
# Modelo 2: DecisionTreeClassifier
# ============================================================================


def entrenar_decision_tree(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    max_depth: int | None = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    class_weight: str | None = None,
    random_state: int = SEED,
    log_en_mlflow: bool = True,
) -> tuple[DecisionTreeClassifier, dict[str, float]]:
    """Entrena un arbol de decision y registra en MLflow.

    El arbol de decision capta relaciones no lineales y umbrales mediante
    una cascada de preguntas binarias sobre las features. A diferencia de
    LogisticRegression, puede captar interacciones entre variables.

    Tiende al overfitting si no se limita su crecimiento, por eso usamos
    hiperparametros que controlan la profundidad y el numero minimo de
    muestras por nodo.

    Args:
        X_train: Features de entrenamiento (ya preprocesadas).
        y_train: Target de entrenamiento.
        X_test: Features de test.
        y_test: Target de test.
        max_depth: Profundidad maxima del arbol. None = sin limite (riesgo
            de overfitting). Defaults to 10.
        min_samples_split: Numero minimo de muestras para dividir un nodo.
            Defaults to 20.
        min_samples_leaf: Numero minimo de muestras en una hoja.
            Defaults to 10.
        class_weight: Si 'balanced', penaliza mas los errores en la clase
            minoritaria. Defaults to None.
        random_state: Semilla para reproducibilidad. Defaults to SEED.
        log_en_mlflow: Si True, registra el experimento en MLflow.
            Defaults to True.

    Returns:
        tuple: (modelo_entrenado, dict_de_metricas).
    """
    nombre_modelo = "DecisionTree"
    if class_weight == "balanced":
        nombre_modelo += "_balanced"

    if log_en_mlflow:
        mlflow.start_run(run_name=nombre_modelo)
        mlflow.log_params(
            {
                "model_type": "DecisionTreeClassifier",
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "class_weight": str(class_weight),
                "random_state": random_state,
            }
        )

    modelo = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
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


# ============================================================================
# Modelo 3: RandomForestClassifier
# ============================================================================


def entrenar_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int | None = 20,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    class_weight: str | None = None,
    n_jobs: int = -1,
    random_state: int = SEED,
    log_en_mlflow: bool = True,
) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Entrena un Random Forest y registra en MLflow.

    Random Forest es un ensemble de muchos arboles de decision entrenados
    sobre muestras aleatorias de filas (bootstrap) y subconjuntos aleatorios
    de features. La prediccion final es el promedio/votacion de todos los
    arboles, lo que reduce el overfitting de un arbol individual y mejora
    la generalizacion.

    Args:
        X_train: Features de entrenamiento (ya preprocesadas).
        y_train: Target de entrenamiento.
        X_test: Features de test.
        y_test: Target de test.
        n_estimators: Numero de arboles en el bosque. Defaults to 100.
        max_depth: Profundidad maxima de cada arbol. Defaults to 20.
        min_samples_split: Minimo de muestras para dividir un nodo.
            Defaults to 10.
        min_samples_leaf: Minimo de muestras en una hoja. Defaults to 5.
        class_weight: Si 'balanced', penaliza mas los errores en la clase
            minoritaria. Defaults to None.
        n_jobs: Numero de cores para paralelizar. -1 usa todos. Defaults to -1.
        random_state: Semilla para reproducibilidad. Defaults to SEED.
        log_en_mlflow: Si True, registra el experimento en MLflow.
            Defaults to True.

    Returns:
        tuple: (modelo_entrenado, dict_de_metricas).
    """
    nombre_modelo = "RandomForest"
    if class_weight == "balanced":
        nombre_modelo += "_balanced"

    if log_en_mlflow:
        mlflow.start_run(run_name=nombre_modelo)
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "class_weight": str(class_weight),
                "random_state": random_state,
            }
        )

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
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


# ============================================================================
# Modelo 4: XGBClassifier (Gradient Boosting)
# ============================================================================


def entrenar_xgboost(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float | None = None,
    n_jobs: int = -1,
    random_state: int = SEED,
    log_en_mlflow: bool = True,
) -> tuple[XGBClassifier, dict[str, float]]:
    """Entrena un modelo XGBoost (Gradient Boosting) y registra en MLflow.

    XGBoost entrena arboles de forma SECUENCIAL: cada arbol nuevo corrige
    los errores del anterior (gradient boosting). A diferencia de Random
    Forest (arboles paralelos e independientes), esta correccion iterativa
    suele dar el mejor rendimiento en datos tabulares.

    Args:
        X_train: Features de entrenamiento (ya preprocesadas).
        y_train: Target de entrenamiento.
        X_test: Features de test.
        y_test: Target de test.
        n_estimators: Numero de arboles (rondas de boosting). Defaults to 300.
        max_depth: Profundidad maxima de cada arbol. Mas bajo que RF porque
            el boosting combina muchos arboles debiles. Defaults to 6.
        learning_rate: Tasa de aprendizaje. Controla cuanto contribuye cada
            arbol. Valores bajos = aprendizaje mas lento pero mas robusto.
            Defaults to 0.1.
        subsample: Fraccion de filas usadas por cada arbol. Introduce
            aleatoriedad para reducir overfitting. Defaults to 0.8.
        colsample_bytree: Fraccion de features usadas por cada arbol.
            Defaults to 0.8.
        scale_pos_weight: Peso de la clase positiva. Para datasets
            desbalanceados, se suele poner ratio negativos/positivos.
            None = sin ajuste. Defaults to None.
        n_jobs: Cores para paralelizar. -1 usa todos. Defaults to -1.
        random_state: Semilla para reproducibilidad. Defaults to SEED.
        log_en_mlflow: Si True, registra el experimento en MLflow.
            Defaults to True.

    Returns:
        tuple: (modelo_entrenado, dict_de_metricas).
    """
    nombre_modelo = "XGBoost"
    if scale_pos_weight is not None:
        nombre_modelo += "_balanced"

    if log_en_mlflow:
        mlflow.start_run(run_name=nombre_modelo)
        mlflow.log_params(
            {
                "model_type": "XGBClassifier",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "scale_pos_weight": str(scale_pos_weight),
                "random_state": random_state,
            }
        )

    modelo = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        n_jobs=n_jobs,
        random_state=random_state,
        eval_metric="logloss",
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


# ============================================================================
# Modelo 5: Red Neuronal con Keras
# ============================================================================


def entrenar_red_neuronal(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    capas_ocultas: tuple[int, ...] = (128, 64, 32),
    dropout: float = 0.3,
    learning_rate: float = 0.001,
    epochs: int = 50,
    batch_size: int = 256,
    paciencia: int = 5,
    random_state: int = SEED,
    log_en_mlflow: bool = True,
) -> tuple[keras.Model, dict[str, float]]:
    """Entrena una red neuronal con Keras y registra en MLflow.

    Red neuronal feed-forward (multicapa) para clasificacion binaria. A
    diferencia de los modelos basados en arboles, la red aprende ajustando
    pesos de neuronas mediante descenso de gradiente a lo largo de varias
    epocas. Necesita features escaladas (ya las tenemos del preprocesamiento).

    Usa EarlyStopping para detener el entrenamiento cuando deja de mejorar,
    evitando overfitting y ahorrando tiempo.

    Args:
        X_train: Features de entrenamiento (ya escaladas).
        y_train: Target de entrenamiento.
        X_test: Features de test.
        y_test: Target de test.
        capas_ocultas: Neuronas en cada capa oculta. Defaults to (128, 64, 32).
        dropout: Fraccion de neuronas que se "apagan" aleatoriamente en cada
            paso, para reducir overfitting. Defaults to 0.3.
        learning_rate: Tasa de aprendizaje del optimizador Adam.
            Defaults to 0.001.
        epochs: Numero maximo de pasadas completas por los datos.
            Defaults to 50.
        batch_size: Numero de muestras por actualizacion de pesos.
            Defaults to 256.
        paciencia: Epocas sin mejora antes de parar (EarlyStopping).
            Defaults to 5.
        random_state: Semilla para reproducibilidad. Defaults to SEED.
        log_en_mlflow: Si True, registra el experimento en MLflow.
            Defaults to True.

    Returns:
        tuple: (modelo_entrenado, dict_de_metricas).
    """
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    n_features = X_train.shape[1]

    if log_en_mlflow:
        mlflow.start_run(run_name="RedNeuronal")
        mlflow.log_params(
            {
                "model_type": "RedNeuronal_Keras",
                "capas_ocultas": str(capas_ocultas),
                "dropout": dropout,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "paciencia": paciencia,
                "random_state": random_state,
            }
        )

    # Construir la arquitectura
    modelo = keras.Sequential(name="red_cancelaciones")
    modelo.add(keras.Input(shape=(n_features,)))

    for n_neuronas in capas_ocultas:
        modelo.add(layers.Dense(n_neuronas, activation="relu"))
        modelo.add(layers.Dropout(dropout))

    modelo.add(layers.Dense(1, activation="sigmoid"))

    # Compilar
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    # EarlyStopping: para si no mejora la val_loss
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=paciencia,
        restore_best_weights=True,
    )

    # Entrenar
    modelo.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    # Predicciones
    y_pred_proba = modelo.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metricas = calcular_metricas(y_test, y_pred, y_pred_proba)

    if log_en_mlflow:
        mlflow.log_metrics(metricas)
        mlflow.tensorflow.log_model(modelo, "model")
        mlflow.end_run()

    return modelo, metricas
