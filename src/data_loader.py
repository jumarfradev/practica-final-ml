"""Módulo de carga y preprocesamiento del dataset de cancelaciones hoteleras.

Este módulo encapsula toda la lógica de preparación de datos para los modelos:

- Carga del dataset desde data/raw/.
- Eliminación de variables con data leakage.
- Limpieza de outliers imposibles.
- Imputación de nulos y creación de flags.
- Pipeline de transformaciones (encoding, escalado).
- Split train/test estratificado.

Uso típico:

    from src.data_loader import cargar_dataset, preparar_datos

    df = cargar_dataset()
    X_train, X_test, y_train, y_test, preprocessor = preparar_datos(df)
"""

from pathlib import Path

import pandas as pd

# ============================================================================
# Constantes del módulo
# ============================================================================

# Detectar raíz del proyecto: este archivo está en src/, así que .parent.parent
# nos da la raíz (.../practica-final-ml/)
PATH_PROYECTO: Path = Path(__file__).resolve().parent.parent
PATH_DATA_RAW: Path = PATH_PROYECTO / "data" / "raw"
PATH_DATA_PROCESSED: Path = PATH_PROYECTO / "data" / "processed"

PATH_DATASET: Path = PATH_DATA_RAW / "dataset_practica_final.csv"

TARGET_COLUMN: str = "is_canceled"
SEED: int = 42


# ============================================================================
# Funciones de carga
# ============================================================================


def cargar_dataset(ruta: Path | None = None) -> pd.DataFrame:
    """Carga el dataset de cancelaciones hoteleras desde CSV.

    Lee el archivo CSV original sin aplicar transformaciones. Las transformaciones
    se realizan en funciones posteriores del módulo.

    Args:
        ruta (Path | None, optional): Ruta al archivo CSV. Si es None, usa la
            constante PATH_DATASET del módulo. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame con 119.390 filas × 32 columnas (versión cruda).

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.

    Example:
        >>> df = cargar_dataset()
        >>> print(df.shape)
        (119390, 32)
    """
    ruta_efectiva = ruta if ruta is not None else PATH_DATASET

    if not ruta_efectiva.exists():
        raise FileNotFoundError(
            f"No se encuentra el dataset en {ruta_efectiva}. "
            f"Asegúrate de que el archivo está en data/raw/."
        )

    df = pd.read_csv(ruta_efectiva)
    return df
