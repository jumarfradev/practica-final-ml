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


# ============================================================================
# Constantes de limpieza
# ============================================================================

COLUMNAS_LEAKAGE: list[str] = [
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
]
"""Variables con data leakage. Identificadas en Sección 6 del EDA.

- reservation_status: estado final de la reserva, conocido DESPUÉS del evento.
- reservation_status_date: fecha del desenlace, posterior a la predicción.
- assigned_room_type: se rellena distinto según si la reserva se completa o no.
"""


# ============================================================================
# Funciones de limpieza
# ============================================================================


def eliminar_columnas_leakage(
    df: pd.DataFrame,
    columnas: list[str] | None = None,
) -> pd.DataFrame:
    """Elimina columnas con data leakage identificadas en el EDA.

    Las variables con leakage contienen información que NO estaría disponible
    en el momento de predecir si una reserva se cancelará. Su eliminación es
    crítica para que el modelo sea realista en producción.

    Args:
        df (pd.DataFrame): DataFrame del que eliminar las columnas.
        columnas (list[str] | None, optional): Lista de columnas a eliminar.
            Si es None, usa la constante COLUMNAS_LEAKAGE del módulo.
            Defaults to None.

    Returns:
        pd.DataFrame: DataFrame sin las columnas eliminadas. Devuelve una
            COPIA del original, no modifica el DataFrame de entrada.

    Example:
        >>> df_limpio = eliminar_columnas_leakage(df)
        >>> "reservation_status" in df_limpio.columns
        False
    """
    columnas_efectivas = columnas if columnas is not None else COLUMNAS_LEAKAGE
    # errors='ignore' permite que la función no falle si alguna columna ya no existe
    return df.drop(columns=columnas_efectivas, errors="ignore").copy()


def eliminar_outliers_imposibles(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con valores imposibles físicamente.

    Esta función filtra outliers que representan errores de captura, NO valores
    legítimos pero raros. Concretamente:

    - `adr < 0`: precio negativo (imposible).
    - `adults > 10`: más de 10 adultos por reserva (probable error de captura).

    NO se eliminan outliers que sean valores reales aunque atípicos
    (ej. lead_time muy alto, previous_cancellations alto, adr alto pero positivo).
    Esos los conservamos para que los modelos los aprovechen.

    Args:
        df (pd.DataFrame): DataFrame con la columna adr y adults.

    Returns:
        pd.DataFrame: DataFrame sin las filas con valores imposibles.

    Example:
        >>> df_filtrado = eliminar_outliers_imposibles(df)
        >>> (df_filtrado["adr"] < 0).sum()
        0
    """
    n_inicial = len(df)

    mascara = (df["adr"] >= 0) & (df["adults"] <= 10)
    df_filtrado = df[mascara].copy()

    n_eliminadas = n_inicial - len(df_filtrado)
    if n_eliminadas > 0:
        print(
            f"Eliminadas {n_eliminadas} filas con valores imposibles "
            f"({n_eliminadas / n_inicial * 100:.3f}% del total)."
        )

    return df_filtrado


def crear_flags_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features binarias indicando presencia/ausencia en agent y company.

    En el EDA (Sección 5) confirmamos que los nulos en agent y company son MNAR
    (Missing Not At Random): su ausencia ES información valiosa.

    - `tiene_agent` = 1 si hay agente, 0 si la reserva es directa.
    - `tiene_company` = 1 si la reserva es corporativa, 0 si no.

    Estas features se añaden ANTES de imputar los valores nulos originales,
    para capturar la información antes de \"perderla\" con la imputación.

    Args:
        df (pd.DataFrame): DataFrame con las columnas agent y company.

    Returns:
        pd.DataFrame: DataFrame con dos columnas nuevas: tiene_agent y tiene_company.

    Example:
        >>> df_con_flags = crear_flags_nulos(df)
        >>> df_con_flags[["tiene_agent", "tiene_company"]].head()
    """
    df = df.copy()
    df["tiene_agent"] = df["agent"].notnull().astype(int)
    df["tiene_company"] = df["company"].notnull().astype(int)
    return df


# ============================================================================
# Función orquestadora de limpieza
# ============================================================================


def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas las transformaciones de limpieza inicial en orden.

    Orden de aplicación:

    1. Eliminar columnas con data leakage (Sección 6 del EDA).
    2. Eliminar outliers imposibles (Sección 3 del EDA).
    3. Crear flags binarios para nulos MNAR (Sección 5 del EDA).

    Esta función NO imputa nulos ni hace encoding ni escalado. Esas
    transformaciones se aplicarán dentro de un Pipeline de sklearn en las
    sub-fases siguientes.

    Args:
        df (pd.DataFrame): DataFrame original tras cargar el CSV.

    Returns:
        pd.DataFrame: DataFrame limpio listo para entrar al Pipeline de sklearn.

    Example:
        >>> df = cargar_dataset()
        >>> df_limpio = limpiar_dataset(df)
        >>> df_limpio.shape
        (~119000, ~31)
    """
    df = eliminar_columnas_leakage(df)
    df = eliminar_outliers_imposibles(df)
    df = crear_flags_nulos(df)
    return df
