"""Modulo de inferencia: carga el modelo entrenado y predice sobre datos crudos.

Este modulo cierra el flujo "desde los datos hasta la inferencia" del enunciado.
Carga el bundle generado por trainer.py (models/best_model.pkl) y expone
funciones para predecir la probabilidad de cancelacion de reservas nuevas,
partiendo de datos CRUDOS (mismas columnas que el dataset original, sin
preprocesar).

La cadena de inferencia reproduce EXACTAMENTE el preprocesamiento de
entrenamiento:
    datos crudos
      -> limpiar_dataset (eliminar leakage, outliers imposibles, crear flags)
      -> aplicar top-N + Other con las categorias CONGELADAS en entrenamiento
      -> ColumnTransformer fiteado (.transform)
      -> modelo.predict_proba

Uso como modulo:
    from src.predictor import cargar_bundle, predecir_proba
    bundle = cargar_bundle()
    proba = predecir_proba(df_crudo, bundle)

Uso como script (predice sobre un CSV de reservas nuevas):
    python -m src.predictor --input ruta/al/csv_nuevo.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import (
    aplicar_top_n_other_con_categorias,
    limpiar_dataset,
)

PATH_PROYECTO: Path = Path(__file__).resolve().parent.parent
PATH_MODELS: Path = PATH_PROYECTO / "models"
PATH_BEST_MODEL: Path = PATH_MODELS / "best_model.pkl"

UMBRAL_DEFECTO: float = 0.5
"""Umbral de decision para convertir probabilidad en clase 0/1."""


def cargar_bundle(ruta: Path | None = None) -> dict:
    """Carga el bundle de inferencia generado por trainer.py.

    Si el modelo ganador era una red neuronal (Keras), la carga tambien el
    modelo .keras y lo inyecta en el bundle bajo la clave 'modelo'.

    Args:
        ruta (Path | None): Ruta al best_model.pkl. Si None, usa la de defecto.

    Returns:
        dict: Bundle con modelo, preprocesador, categorias_top_n, etc.

    Raises:
        FileNotFoundError: Si no existe el bundle (no se ha entrenado aun).
    """
    ruta_efectiva = ruta if ruta is not None else PATH_BEST_MODEL

    if not ruta_efectiva.exists():
        raise FileNotFoundError(
            f"No se encuentra el modelo entrenado en {ruta_efectiva}. "
            f"Ejecuta primero 'python trainer.py' para generarlo."
        )

    bundle = joblib.load(ruta_efectiva)

    # Si el ganador es Keras, cargar el modelo .keras aparte
    if bundle.get("es_keras") and bundle.get("modelo") is None:
        from tensorflow import keras

        ruta_keras = PATH_MODELS / bundle["ruta_keras"]
        bundle["modelo"] = keras.models.load_model(ruta_keras)

    return bundle


def _preparar_datos_crudos(df_crudo: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Reproduce la cadena de preprocesamiento de train sobre datos crudos.

    Args:
        df_crudo: DataFrame con las columnas originales del dataset (sin target
            necesariamente). Puede ser una o varias filas.
        bundle: Bundle cargado con preprocesador y categorias congeladas.

    Returns:
        np.ndarray: Matriz de features lista para el modelo.
    """
    # 1. Limpieza (elimina leakage si existiese, outliers imposibles, crea flags)
    df = limpiar_dataset(df_crudo)

    # 2. Top-N + Other con las categorias CONGELADAS en entrenamiento
    df = aplicar_top_n_other_con_categorias(df, bundle["categorias_top_n"])

    # 3. ColumnTransformer fiteado
    X = bundle["preprocesador"].transform(df)

    return X


def predecir_proba(df_crudo: pd.DataFrame, bundle: dict | None = None) -> np.ndarray:
    """Predice la probabilidad de cancelacion para reservas crudas.

    Args:
        df_crudo: DataFrame con reservas en formato crudo (columnas originales).
        bundle: Bundle ya cargado. Si None, lo carga de disco.

    Returns:
        np.ndarray: Array de probabilidades de cancelacion (clase 1), una por fila.
    """
    if bundle is None:
        bundle = cargar_bundle()

    X = _preparar_datos_crudos(df_crudo, bundle)
    modelo = bundle["modelo"]

    if bundle.get("es_keras"):
        proba = modelo.predict(X, verbose=0).ravel()
    else:
        proba = modelo.predict_proba(X)[:, 1]

    return proba


def predecir(
    df_crudo: pd.DataFrame,
    bundle: dict | None = None,
    umbral: float = UMBRAL_DEFECTO,
) -> pd.DataFrame:
    """Predice clase y probabilidad de cancelacion para reservas crudas.

    Args:
        df_crudo: DataFrame con reservas en formato crudo.
        bundle: Bundle ya cargado. Si None, lo carga de disco.
        umbral: Umbral para convertir probabilidad en clase. Defaults to 0.5.

    Returns:
        pd.DataFrame: DataFrame con columnas 'probabilidad_cancelacion' y
            'prediccion' (0 = no cancela, 1 = cancela).
    """
    if bundle is None:
        bundle = cargar_bundle()

    proba = predecir_proba(df_crudo, bundle)
    pred = (proba >= umbral).astype(int)

    return pd.DataFrame(
        {
            "probabilidad_cancelacion": proba,
            "prediccion": pred,
        }
    )


def _main() -> None:
    """Punto de entrada CLI: predice sobre un CSV de reservas nuevas."""
    parser = argparse.ArgumentParser(
        description="Predice la probabilidad de cancelacion de reservas nuevas "
        "usando el modelo entrenado (models/best_model.pkl)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta a un CSV con reservas en formato crudo (columnas originales).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta de salida para el CSV con predicciones. Si se omite, "
        "imprime las primeras filas por consola.",
    )
    parser.add_argument(
        "--umbral",
        type=float,
        default=UMBRAL_DEFECTO,
        help=f"Umbral de decision. Defaults to {UMBRAL_DEFECTO}.",
    )
    args = parser.parse_args()

    ruta_input = Path(args.input)
    if not ruta_input.exists():
        print(f"FAIL: no existe el archivo {ruta_input}", file=sys.stderr)
        sys.exit(1)

    bundle = cargar_bundle()
    print(
        f"Modelo cargado: {bundle['nombre_modelo']} "
        f"(ROC-AUC entrenamiento: {bundle['metricas'].get('roc_auc', 0):.4f})"
    )

    df = pd.read_csv(ruta_input)
    resultado = predecir(df, bundle, umbral=args.umbral)

    salida = pd.concat([df.reset_index(drop=True), resultado], axis=1)

    if args.output:
        salida.to_csv(args.output, index=False)
        print(f"OK: predicciones guardadas en {args.output}")
    else:
        print(salida.head(10).to_string())


if __name__ == "__main__":
    _main()
