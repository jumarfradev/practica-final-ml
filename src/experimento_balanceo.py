"""Experimento: impacto del balanceo de clases en el dataset (63/37).

OBJETIVO
--------
Demostrar empiricamente como afecta balancear el dataset al rendimiento del
modelo, y justificar la decision de NO balancear en este proyecto.

Compara el MISMO modelo (RandomForest) en tres configuraciones identicas salvo
el tratamiento del desbalanceo:

    1. SIN balanceo        -> baseline (configuracion actual del proyecto).
    2. class_weight=balanced -> reponderacion: penaliza mas los errores en la
       clase minoritaria (cancelaciones).
    3. SMOTE               -> sobremuestreo sintetico: genera ejemplos
       artificiales de la clase minoritaria SOLO en entrenamiento.

IDEA CLAVE QUE SE DEMUESTRA
---------------------------
Con un desbalanceo moderado (63/37), balancear no "mejora" el modelo de forma
global: mueve el equilibrio entre recall (sube: detecta mas cancelaciones) y
precision (baja: mas falsas alarmas), mientras que ROC-AUC apenas cambia. Como
el modelo sin balancear ya alcanza un F1 alto, el balanceo no compensa.

RIGOR METODOLOGICO
------------------
SMOTE se aplica EXCLUSIVAMENTE sobre el conjunto de entrenamiento, nunca sobre
test. Generar ejemplos sinteticos y evaluarlos seria data leakage y daria
metricas infladas y falsas. Por eso se aplica tras el split, solo en train.

Uso:
    python -m src.experimento_balanceo
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import SEED, preparar_datos
from src.models import calcular_metricas

PATH_PROYECTO: Path = Path(__file__).resolve().parent.parent
PATH_FIGURAS: Path = PATH_PROYECTO / "resultados" / "figuras"
PATH_METRICAS: Path = PATH_PROYECTO / "resultados" / "metricas"

# Hiperparametros FIJOS para las tres variantes (solo cambia el balanceo).
# Se usan los del RandomForest base para que la comparacion sea limpia.
HIPERPARAMS_RF: dict = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": SEED,
}

ORDEN_METRICAS: list[str] = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _entrenar_y_evaluar(
    modelo: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> dict[str, float]:
    """Entrena el modelo y devuelve todas las metricas en test."""
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    return calcular_metricas(y_test, y_pred, y_pred_proba)


def ejecutar_experimento() -> pd.DataFrame:
    """Ejecuta las tres variantes de balanceo y devuelve la tabla comparativa.

    Returns:
        pd.DataFrame: Filas = configuraciones, columnas = metricas.
    """
    print("Preparando datos...")
    X_train, X_test, y_train, y_test, _ = preparar_datos()

    # Proporcion real de clases en train (para documentar el desbalanceo)
    prop = y_train.value_counts(normalize=True)
    print(
        f"Distribucion train -> no cancela: {prop.get(0, 0) * 100:.1f}%, "
        f"cancela: {prop.get(1, 0) * 100:.1f}%"
    )

    resultados: dict[str, dict[str, float]] = {}

    # --- 1. SIN balanceo (baseline) ---
    print("\n[1/3] RandomForest SIN balanceo...")
    modelo = RandomForestClassifier(**HIPERPARAMS_RF)
    resultados["Sin balanceo"] = _entrenar_y_evaluar(modelo, X_train, y_train, X_test, y_test)

    # --- 2. class_weight balanced ---
    print("[2/3] RandomForest con class_weight=balanced...")
    modelo = RandomForestClassifier(class_weight="balanced", **HIPERPARAMS_RF)
    resultados["class_weight"] = _entrenar_y_evaluar(modelo, X_train, y_train, X_test, y_test)

    # --- 3. SMOTE (solo en train) ---
    print("[3/3] RandomForest con SMOTE (sobremuestreo solo en train)...")
    smote = SMOTE(random_state=SEED)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    prop_smote = pd.Series(y_train_smote).value_counts(normalize=True)
    print(
        f"  Tras SMOTE -> no cancela: {prop_smote.get(0, 0) * 100:.1f}%, "
        f"cancela: {prop_smote.get(1, 0) * 100:.1f}% (balanceado al 50/50)"
    )
    modelo = RandomForestClassifier(**HIPERPARAMS_RF)
    resultados["SMOTE"] = _entrenar_y_evaluar(modelo, X_train_smote, y_train_smote, X_test, y_test)

    # Construir tabla ordenada
    tabla = pd.DataFrame(resultados).T[ORDEN_METRICAS]
    return tabla


def graficar_comparativa(tabla: pd.DataFrame, guardar: bool = True) -> None:
    """Grafica de barras agrupadas comparando metricas entre configuraciones."""
    PATH_FIGURAS.mkdir(parents=True, exist_ok=True)

    metricas = ORDEN_METRICAS
    configs = list(tabla.index)
    x = np.arange(len(metricas))
    ancho = 0.25
    colores = ["#2E75B6", "#ED7D31", "#70AD47"]

    plt.figure(figsize=(12, 7))
    for i, config in enumerate(configs):
        valores = [tabla.loc[config, m] for m in metricas]
        barras = plt.bar(x + i * ancho, valores, ancho, label=config, color=colores[i % 3])
        for barra, valor in zip(barras, valores, strict=True):
            plt.text(
                barra.get_x() + barra.get_width() / 2,
                valor + 0.005,
                f"{valor:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xticks(x + ancho, [m.upper() for m in metricas])
    plt.ylabel("Valor de la metrica", fontsize=12)
    plt.title(
        "Impacto del balanceo de clases (RandomForest, desbalanceo 63/37)",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylim(0, 1.05)
    plt.legend(title="Configuracion", fontsize=10)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    if guardar:
        ruta = PATH_FIGURAS / "comparativa_balanceo.png"
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"\nFigura guardada: {ruta}")

    plt.show()


def imprimir_conclusion(tabla: pd.DataFrame) -> None:
    """Imprime una interpretacion automatica de los resultados."""
    base = tabla.loc["Sin balanceo"]

    print("\n" + "=" * 70)
    print("TABLA COMPARATIVA")
    print("=" * 70)
    print(tabla.round(4).to_string())

    print("\n" + "=" * 70)
    print("INTERPRETACION")
    print("=" * 70)
    for config in tabla.index:
        if config == "Sin balanceo":
            continue
        d_auc = tabla.loc[config, "roc_auc"] - base["roc_auc"]
        d_recall = tabla.loc[config, "recall"] - base["recall"]
        d_prec = tabla.loc[config, "precision"] - base["precision"]
        d_f1 = tabla.loc[config, "f1"] - base["f1"]
        print(
            f"\n{config} (vs sin balanceo):"
            f"\n  ROC-AUC : {d_auc:+.4f}"
            f"\n  Recall  : {d_recall:+.4f}"
            f"\n  Precision: {d_prec:+.4f}"
            f"\n  F1      : {d_f1:+.4f}"
        )

    print("\n" + "=" * 70)
    print(
        "CONCLUSION: con desbalanceo moderado (63/37), el balanceo apenas mueve\n"
        "ROC-AUC y desplaza el equilibrio recall/precision. Si la mejora de F1\n"
        "no es relevante, se justifica NO balancear y entrenar sobre la\n"
        "distribucion real."
    )
    print("=" * 70)


def main() -> None:
    """Ejecuta el experimento completo: tabla + grafica + conclusion."""
    tabla = ejecutar_experimento()
    imprimir_conclusion(tabla)
    graficar_comparativa(tabla)

    # Guardar la tabla en CSV para el informe
    PATH_METRICAS.mkdir(parents=True, exist_ok=True)
    ruta_csv = PATH_METRICAS / "comparativa_balanceo.csv"
    tabla.round(4).to_csv(ruta_csv)
    print(f"\nTabla guardada: {ruta_csv}")


if __name__ == "__main__":
    main()
