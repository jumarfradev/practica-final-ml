"""Comparador multi-modelo: guarda los 8 modelos y predice con todos a la vez.

Mientras que best_model.pkl contiene SOLO el modelo ganador (el que se sirve en
produccion), este modulo permite guardar TODOS los modelos entrenados y, dada
una reserva, obtener la prediccion de cada uno. Sirve para la vista comparativa
del Streamlit: "esta reserva la predice el ganador, pero asi se comportarian los
demas modelos".

IMPORTANTE: en produccion real se sirve un unico modelo (el mejor). Esta
comparativa tiene fines DEMOSTRATIVOS y didacticos, no de produccion.

Los modelos Keras (red neuronal) no se serializan con joblib; se excluyen del
bundle comparativo y se marcan como no disponibles en la comparativa.

Estructura de all_models.pkl:
    {
      "preprocesador": ColumnTransformer ajustado (compartido),
      "categorias_top_n": dict,
      "nombres_features": list,
      "modelos": { "RandomForest_OPT": modelo, "XGBoost": modelo, ... },
      "metricas": { "RandomForest_OPT": {...}, ... },
    }
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import aplicar_top_n_other_con_categorias, limpiar_dataset

PATH_PROYECTO: Path = Path(__file__).resolve().parent.parent
PATH_MODELS: Path = PATH_PROYECTO / "models"
PATH_ALL_MODELS: Path = PATH_MODELS / "all_models.pkl"

# Modelos Keras que no se serializan con joblib (se excluyen de la comparativa).
NOMBRES_KERAS: set[str] = {"RedNeuronal", "RedNeuronal_OPT"}


def guardar_todos_los_modelos(
    resultados: dict[str, tuple],
    preprocesador: object,
    categorias_top_n: dict[str, list],
    nombres_features: list[str],
) -> None:
    """Guarda todos los modelos sklearn/xgboost en all_models.pkl.

    Args:
        resultados: Diccionario {nombre: (modelo, metricas)} de trainer.
        preprocesador: ColumnTransformer ajustado.
        categorias_top_n: Categorias top-N congeladas.
        nombres_features: Nombres de las features de salida.
    """
    PATH_MODELS.mkdir(parents=True, exist_ok=True)

    modelos = {}
    metricas = {}
    for nombre, (modelo, met) in resultados.items():
        if nombre in NOMBRES_KERAS:
            continue  # Keras no se serializa con joblib
        modelos[nombre] = modelo
        metricas[nombre] = met

    bundle = {
        "preprocesador": preprocesador,
        "categorias_top_n": categorias_top_n,
        "nombres_features": nombres_features,
        "modelos": modelos,
        "metricas": metricas,
    }
    joblib.dump(bundle, PATH_ALL_MODELS)
    print(f"Bundle comparativo guardado en: {PATH_ALL_MODELS} ({len(modelos)} modelos)")


def cargar_todos_los_modelos(ruta: Path | None = None) -> dict:
    """Carga el bundle comparativo con todos los modelos.

    Raises:
        FileNotFoundError: Si no existe (no se ha entrenado en modo completo).
    """
    ruta_efectiva = ruta if ruta is not None else PATH_ALL_MODELS
    if not ruta_efectiva.exists():
        raise FileNotFoundError(
            f"No se encuentra {ruta_efectiva}. Ejecuta 'python trainer.py' "
            f"(completo) para generar el bundle comparativo."
        )
    return joblib.load(ruta_efectiva)


def _preparar(df_crudo: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Reproduce el preprocesamiento sobre datos crudos (compartido por todos)."""
    df = limpiar_dataset(df_crudo)
    df = aplicar_top_n_other_con_categorias(df, bundle["categorias_top_n"])
    return bundle["preprocesador"].transform(df)


def predecir_con_todos(df_crudo: pd.DataFrame, bundle: dict | None = None) -> list[dict]:
    """Predice la probabilidad de cancelacion de UNA reserva con todos los modelos.

    Args:
        df_crudo: DataFrame con UNA reserva en formato crudo.
        bundle: Bundle comparativo. Si None, lo carga de disco.

    Returns:
        list[dict]: Una entrada por modelo con su nombre, probabilidad, prediccion
            y ROC-AUC global, ordenadas por ROC-AUC descendente.
    """
    if bundle is None:
        bundle = cargar_todos_los_modelos()

    X = _preparar(df_crudo, bundle)

    resultados = []
    for nombre, modelo in bundle["modelos"].items():
        proba = float(modelo.predict_proba(X)[:, 1][0])
        met = bundle["metricas"].get(nombre, {})
        resultados.append(
            {
                "modelo": nombre,
                "probabilidad_cancelacion": round(proba, 4),
                "prediccion": int(proba >= 0.5),
                "roc_auc_global": round(met.get("roc_auc", 0), 4),
            }
        )

    # Ordenar por ROC-AUC global (mejor modelo primero)
    resultados.sort(key=lambda r: r["roc_auc_global"], reverse=True)
    return resultados
