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

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

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


# ============================================================================
# Constantes para el preprocesamiento (sub-fase 3.3)
# ============================================================================

NUMERICAS_REALES: list[str] = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]
"""Variables numéricas reales. Reciben imputación con mediana + StandardScaler."""

NUMERICAS_LOG: list[str] = [
    "lead_time",
    "adr",
]
"""Subconjunto de NUMERICAS_REALES con asimetría alta (|skew| > 1) en el EDA.
Se les aplica transformación log1p antes del escalado."""

NUMERICAS_PASS_THROUGH: list[str] = [
    "tiene_agent",
    "tiene_company",
    "is_repeated_guest",
]
"""Variables binarias (0/1). Pasan sin transformar al modelo. No requieren
imputación (los flags no tienen nulos) ni escalado (ya están en [0, 1])."""

CATEGORICAS_BAJA_CARDINALIDAD: list[str] = [
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "deposit_type",
    "customer_type",
    "reserved_room_type",
    "arrival_date_month",
]
"""Categóricas con ≤12 categorías únicas. Reciben OneHotEncoder directo."""

CATEGORICAS_ALTA_CARDINALIDAD: list[str] = [
    "country",
    "agent",
]
"""Categóricas con muchos valores únicos. Necesitan reducción top-N + Other
antes del OneHotEncoder. company se ha eliminado (solo conservamos el flag)."""

COLUMNAS_TEMPORALES: list[str] = [
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
]
"""Variables temporales numéricas. Reciben StandardScaler simple por ahora.
Mejora opcional: codificación cíclica con seno/coseno para week_number y month."""

# ============================================================================
# Parámetros del preprocesador
# ============================================================================

TOP_N_COUNTRIES: int = 30
"""Número de países más frecuentes a conservar. El resto se agrupa en 'Other'."""

TOP_N_AGENTS: int = 30
"""Número de IDs de agente más frecuentes a conservar. El resto se agrupa en 'Other'."""
# ============================================================================
# Funciones auxiliares para el preprocesador
# ============================================================================


def _agrupar_categorias_raras(df: pd.DataFrame, columna: str, top_n: int) -> pd.Series:
    """Reemplaza las categorías poco frecuentes por 'Other'.

    Conserva las top_n categorías más frecuentes y agrupa el resto bajo 'Other'.
    Útil para variables de alta cardinalidad como country o agent, donde aplicar
    OneHotEncoder directo generaría cientos de columnas casi vacías.

    Args:
        df (pd.DataFrame): DataFrame con la columna.
        columna (str): Nombre de la columna a transformar.
        top_n (int): Número de categorías más frecuentes a conservar.

    Returns:
        pd.Series: Columna transformada con 'Other' donde corresponda.

    Example:
        >>> _agrupar_categorias_raras(df, "country", top_n=30)
    """
    top_categorias = df[columna].value_counts().head(top_n).index
    return df[columna].where(df[columna].isin(top_categorias), other="Other")


def aplicar_top_n_other(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la reducción top-N + 'Other' a variables de alta cardinalidad.

    Transforma in-place las columnas de CATEGORICAS_ALTA_CARDINALIDAD para
    reducir su cardinalidad antes de pasar por el ColumnTransformer.

    - country: conserva top TOP_N_COUNTRIES, resto → 'Other'.
    - agent: conserva top TOP_N_AGENTS, resto → 'Other'. Los nulos se convierten
      en string 'sin_agent' (que tras top_n probablemente quede en 'Other').

    Args:
        df (pd.DataFrame): DataFrame del que reducir cardinalidad.

    Returns:
        pd.DataFrame: DataFrame con columnas de alta cardinalidad agrupadas.
    """
    df = df.copy()

    # country: top-N + Other (los nulos ya se rellenarán con SimpleImputer después)
    df["country"] = _agrupar_categorias_raras(df, "country", TOP_N_COUNTRIES)

    # agent: convertir IDs numéricos a string, agrupar top-N + Other
    # Los nulos se convierten a string para que entren al pipeline categórico
    df["agent"] = df["agent"].fillna("sin_agent").astype(str)
    df["agent"] = _agrupar_categorias_raras(df, "agent", TOP_N_AGENTS)

    return df


# ============================================================================
# Construcción del preprocesador (ColumnTransformer)
# ============================================================================


def construir_preprocesador() -> ColumnTransformer:
    """Construye el preprocesador completo del proyecto.

    Devuelve un ColumnTransformer de sklearn que aplica las transformaciones
    apropiadas a cada grupo de columnas. Es la pieza central del pipeline de
    preprocesamiento.

    Pipelines internos:

    - Numéricas con log: SimpleImputer(median) → log1p → StandardScaler.
    - Numéricas normales: SimpleImputer(median) → StandardScaler.
    - Categóricas baja card.: SimpleImputer(most_frequent) → OneHotEncoder.
    - Categóricas alta card.: SimpleImputer(most_frequent) → OneHotEncoder
      (asume que aplicar_top_n_other() ya se aplicó antes).
    - Pass-through (binarias): sin transformación.
    - Temporales: SimpleImputer(median) → StandardScaler.

    Returns:
        ColumnTransformer: Preprocesador listo para fit() y transform().

    Example:
        >>> preprocesador = construir_preprocesador()
        >>> preprocesador.fit_transform(X_train)
    """
    # Pipeline numérico con transformación logarítmica
    pipeline_numerico_log = Pipeline(
        [
            ("imputador", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("escalado", StandardScaler()),
        ]
    )

    # Pipeline numérico estándar (sin log)
    pipeline_numerico = Pipeline(
        [
            ("imputador", SimpleImputer(strategy="median")),
            ("escalado", StandardScaler()),
        ]
    )

    # Pipeline categórico
    pipeline_categorico = Pipeline(
        [
            ("imputador", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Numéricas que no necesitan log son las que están en NUMERICAS_REALES pero NO en NUMERICAS_LOG
    numericas_sin_log = [c for c in NUMERICAS_REALES if c not in NUMERICAS_LOG]

    # Ensamblar el ColumnTransformer
    preprocesador = ColumnTransformer(
        transformers=[
            ("num_log", pipeline_numerico_log, NUMERICAS_LOG),
            ("num", pipeline_numerico, numericas_sin_log),
            ("cat_baja", pipeline_categorico, CATEGORICAS_BAJA_CARDINALIDAD),
            ("cat_alta", pipeline_categorico, CATEGORICAS_ALTA_CARDINALIDAD),
            ("temporales", pipeline_numerico, COLUMNAS_TEMPORALES),
            ("passthrough", "passthrough", NUMERICAS_PASS_THROUGH),
        ],
        remainder="drop",  # Cualquier columna no listada se descarta
        verbose_feature_names_out=False,
    )

    return preprocesador


# ============================================================================
# Funcion orquestadora del preprocesamiento (sin split)
# ============================================================================


def aplicar_preprocesamiento_completo(
    df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica todas las transformaciones previas al split train/test.

    Esta funcion ejecuta el flujo completo de preprocesamiento desde el
    DataFrame crudo hasta tener X (features) e y (target) listos para
    dividirse en train/test.

    Pasos aplicados:
    1. Cargar dataset (si no se pasa uno).
    2. Limpiar (eliminar leakage, outliers, crear flags).
    3. Aplicar top-N + Other a categoricas de alta cardinalidad.
    4. Separar X e y.

    El ColumnTransformer (escalado, encoding) NO se aplica aqui. Se aplica
    despues del split, dentro de un Pipeline, para evitar data leakage de
    train a test.

    Args:
        df (pd.DataFrame | None, optional): DataFrame con los datos crudos.
            Si es None, carga el dataset con cargar_dataset(). Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Tupla (X, y) donde:
            - X: DataFrame con las features listas para el ColumnTransformer.
            - y: Serie con el target binario is_canceled.

    Example:
        >>> X, y = aplicar_preprocesamiento_completo()
        >>> X.shape, y.shape
        ((119377, 30), (119377,))
    """
    if df is None:
        df = cargar_dataset()

    df = limpiar_dataset(df)
    df = aplicar_top_n_other(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


# ============================================================================
# Split train/test estratificado
# ============================================================================


def split_train_test_estratificado(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide los datos en train y test manteniendo la proporcion del target.

    Usa stratify=y para asegurar que la proporcion de cancelaciones (63/37)
    se mantenga exactamente igual en train y en test. Esto es crucial en
    clasificacion binaria con clases desbalanceadas: un split aleatorio
    podria producir conjuntos con distribuciones distintas, sesgando las
    metricas.

    Args:
        X (pd.DataFrame): Features ya preprocesadas (sin ColumnTransformer aun).
        y (pd.Series): Target binario is_canceled.
        test_size (float, optional): Proporcion del test set. Defaults to 0.2.
        random_state (int, optional): Semilla para reproducibilidad.
            Defaults to SEED (42).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Tupla
            (X_train, X_test, y_train, y_test).

    Example:
        >>> X_train, X_test, y_train, y_test = split_train_test_estratificado(X, y)
        >>> X_train.shape, X_test.shape
        ((95501, 30), (23876, 30))
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


# ============================================================================
# Funcion orquestadora final del preprocesamiento
# ============================================================================


def preparar_datos(
    test_size: float = 0.2,
    random_state: int = SEED,
    devolver_dataframes: bool = False,
) -> tuple:
    """Ejecuta el pipeline completo de preprocesamiento.

    Esta es la funcion principal del modulo. Encapsula todo el flujo desde
    cargar el CSV hasta tener X_train, X_test, y_train, y_test listos para
    entrenar modelos.

    Orden de operaciones:
    1. Cargar el dataset.
    2. Limpiar (eliminar leakage, outliers, crear flags).
    3. Aplicar top-N + Other a country y agent.
    4. Separar X e y.
    5. Split train/test estratificado.
    6. Construir el ColumnTransformer.
    7. fit_transform en X_train (aprende escalado/encoding del train).
    8. transform en X_test (aplica lo aprendido sin filtrar informacion).

    Args:
        test_size (float, optional): Proporcion del test set. Defaults to 0.2.
        random_state (int, optional): Semilla para reproducibilidad.
            Defaults to SEED (42).
        devolver_dataframes (bool, optional): Si True, X_train y X_test se
            devuelven como DataFrames con nombres de columna. Si False,
            como arrays numpy. Defaults to False.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocesador) donde el
            preprocesador es un ColumnTransformer ya entrenado (fit aplicado
            solo en train) listo para usar en datos nuevos con .transform().

    Example:
        >>> X_train, X_test, y_train, y_test, preprocesador = preparar_datos()
        >>> X_train.shape
        (95501, 130)
    """
    X, y = aplicar_preprocesamiento_completo()

    X_train, X_test, y_train, y_test = split_train_test_estratificado(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocesador = construir_preprocesador()

    X_train_transformado = preprocesador.fit_transform(X_train)
    X_test_transformado = preprocesador.transform(X_test)

    if devolver_dataframes:
        feature_names = preprocesador.get_feature_names_out()
        X_train_transformado = pd.DataFrame(
            X_train_transformado, columns=feature_names, index=X_train.index
        )
        X_test_transformado = pd.DataFrame(
            X_test_transformado, columns=feature_names, index=X_test.index
        )

    return X_train_transformado, X_test_transformado, y_train, y_test, preprocesador
