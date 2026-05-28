"""Modulo de persistencia de metricas.

Guarda y carga las metricas de los modelos en disco (JSON), para que la tabla
comparativa sea robusta a reinicios del kernel. Cada modelo guarda sus metricas
en un archivo bajo resultados/metricas/.
"""

import json
from pathlib import Path

PATH_PROYECTO: Path = Path(__file__).resolve().parent.parent
PATH_METRICAS: Path = PATH_PROYECTO / "resultados" / "metricas"


def guardar_metricas(nombre_modelo: str, metricas: dict[str, float]) -> None:
    """Guarda las metricas de un modelo en disco como JSON.

    Args:
        nombre_modelo: Identificador del modelo (sera el nombre del archivo).
        metricas: Diccionario de metricas a guardar.
    """
    PATH_METRICAS.mkdir(parents=True, exist_ok=True)
    ruta = PATH_METRICAS / f"{nombre_modelo}.json"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2)
    print(f"Metricas guardadas: {ruta.name}")


def cargar_metricas(nombre_modelo: str) -> dict[str, float]:
    """Carga las metricas de un modelo desde disco.

    Args:
        nombre_modelo: Identificador del modelo.

    Returns:
        dict: Diccionario de metricas.
    """
    ruta = PATH_METRICAS / f"{nombre_modelo}.json"
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def cargar_todas_las_metricas() -> dict[str, dict[str, float]]:
    """Carga las metricas de TODOS los modelos guardados en disco.

    Returns:
        dict: Diccionario {nombre_modelo: {metricas}} de todos los JSON
            encontrados en la carpeta de metricas.
    """
    if not PATH_METRICAS.exists():
        return {}

    resultados = {}
    for ruta in sorted(PATH_METRICAS.glob("*.json")):
        nombre = ruta.stem
        with open(ruta, encoding="utf-8") as f:
            resultados[nombre] = json.load(f)
    return resultados
