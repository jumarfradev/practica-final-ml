"""Modulo de evaluacion visual de modelos.

Genera las visualizaciones de la Fase 5: curvas ROC comparativas, matrices de
confusion, feature importance y grafico de barras de metricas. Separado de
models.py y optimization.py por responsabilidad unica: este modulo solo
EVALUA y VISUALIZA modelos ya entrenados.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
)

PATH_PROYECTO: Path = Path(__file__).resolve().parent.parent
PATH_FIGURAS: Path = PATH_PROYECTO / "resultados" / "figuras"


def _asegurar_carpeta_figuras() -> None:
    """Crea la carpeta de figuras si no existe."""
    PATH_FIGURAS.mkdir(parents=True, exist_ok=True)


def graficar_curvas_roc(
    modelos_probas: dict[str, np.ndarray],
    y_test: pd.Series | np.ndarray,
    guardar: bool = True,
) -> None:
    """Dibuja las curvas ROC de varios modelos en un mismo grafico.

    La curva ROC representa la tasa de verdaderos positivos frente a la de
    falsos positivos para distintos umbrales. Cuanto mas cerca de la esquina
    superior izquierda, mejor. El area bajo la curva (AUC) es la metrica.

    Args:
        modelos_probas: Diccionario {nombre_modelo: probabilidades_predichas}.
            Las probabilidades son las de la clase positiva (predict_proba[:, 1]).
        y_test: Etiquetas reales del conjunto de test.
        guardar: Si True, guarda la figura en resultados/figuras/. Defaults to True.
    """
    plt.figure(figsize=(10, 8))

    for nombre, y_proba in modelos_probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = np.trapz(tpr, fpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{nombre} (AUC = {auc:.4f})")

    # Linea diagonal de referencia (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatorio (AUC = 0.5)")

    plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=12)
    plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)", fontsize=12)
    plt.title("Curvas ROC comparativas", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if guardar:
        _asegurar_carpeta_figuras()
        ruta = PATH_FIGURAS / "curvas_roc.png"
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {ruta}")

    plt.show()


def graficar_matriz_confusion(
    y_test: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    nombre_modelo: str = "",
    guardar: bool = True,
) -> None:
    """Dibuja la matriz de confusion de un modelo.

    La matriz muestra las predicciones correctas e incorrectas separadas por
    clase: verdaderos negativos, falsos positivos, falsos negativos y
    verdaderos positivos.

    Args:
        y_test: Etiquetas reales.
        y_pred: Predicciones del modelo (0/1).
        nombre_modelo: Nombre para el titulo.
        guardar: Si True, guarda la figura. Defaults to True.
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No cancela", "Cancela"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Matriz de confusion: {nombre_modelo}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if guardar:
        _asegurar_carpeta_figuras()
        nombre_archivo = f"matriz_confusion_{nombre_modelo.replace(' ', '_')}.png"
        ruta = PATH_FIGURAS / nombre_archivo
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {ruta}")

    plt.show()


def graficar_feature_importance(
    modelo: object,
    nombres_features: list[str],
    top_n: int = 20,
    nombre_modelo: str = "",
    guardar: bool = True,
) -> pd.DataFrame:
    """Dibuja la importancia de las features de un modelo basado en arboles.

    Solo funciona con modelos que tengan el atributo feature_importances_
    (RandomForest, XGBoost, DecisionTree).

    Args:
        modelo: Modelo entrenado con atributo feature_importances_.
        nombres_features: Lista de nombres de las features (columnas de X).
        top_n: Numero de features mas importantes a mostrar. Defaults to 20.
        nombre_modelo: Nombre para el titulo.
        guardar: Si True, guarda la figura. Defaults to True.

    Returns:
        pd.DataFrame: Tabla con las features ordenadas por importancia.
    """
    importancias = modelo.feature_importances_

    df_imp = pd.DataFrame({"feature": nombres_features, "importancia": importancias}).sort_values(
        "importancia", ascending=False
    )

    top = df_imp.head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=top, y="feature", x="importancia", color="steelblue")
    plt.xlabel("Importancia", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(
        f"Top {top_n} features mas importantes: {nombre_modelo}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if guardar:
        _asegurar_carpeta_figuras()
        ruta = PATH_FIGURAS / f"feature_importance_{nombre_modelo.replace(' ', '_')}.png"
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {ruta}")

    plt.show()

    return df_imp


def graficar_barras_roc_auc(
    metricas: dict[str, dict[str, float]],
    guardar: bool = True,
) -> None:
    """Dibuja un grafico de barras comparando el ROC-AUC de todos los modelos.

    Args:
        metricas: Diccionario {nombre_modelo: {metricas}}. Se usa la clave
            'roc_auc' de cada modelo.
        guardar: Si True, guarda la figura. Defaults to True.
    """
    nombres = list(metricas.keys())
    valores = [metricas[n]["roc_auc"] for n in nombres]

    # Ordenar de mayor a menor
    orden = np.argsort(valores)[::-1]
    nombres = [nombres[i] for i in orden]
    valores = [valores[i] for i in orden]

    plt.figure(figsize=(11, 7))
    colores = ["#2E75B6" if i > 0 else "#548235" for i in range(len(nombres))]
    barras = plt.barh(nombres[::-1], valores[::-1], color=colores[::-1])

    plt.xlabel("ROC-AUC", fontsize=12)
    plt.title("Comparativa de modelos por ROC-AUC", fontsize=14, fontweight="bold")
    plt.xlim(0.85, 0.97)

    # Etiquetas de valor en cada barra
    for barra, valor in zip(barras, valores[::-1], strict=False):
        plt.text(
            valor + 0.001,
            barra.get_y() + barra.get_height() / 2,
            f"{valor:.4f}",
            va="center",
            fontsize=9,
        )

    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()

    if guardar:
        _asegurar_carpeta_figuras()
        ruta = PATH_FIGURAS / "barras_roc_auc.png"
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {ruta}")

    plt.show()
