"""Analisis del umbral de decision optimo segun coste de negocio.

OBJETIVO
--------
El umbral por defecto de 0.5 trata por igual los dos tipos de error. Pero en el
negocio hotelero NO cuestan lo mismo:

- Falso negativo (predecir "no cancela" cuando SI cancela): el hotel no actua,
  la cancelacion llega tarde para revender la habitacion -> NOCHE PERDIDA, se
  pierde el ADR completo. Error CARO.
- Falso positivo (predecir "cancela" cuando NO cancela): el hotel hace una
  accion de retencion (llamada, email, incentivo menor) que sobra. Error BARATO.

Por eso se asume que un FN cuesta del orden de COSTE_FN / COSTE_FP veces mas que
un FP (por defecto 5:1). Bajo este supuesto, el umbral optimo NO es 0.5: conviene
un umbral mas bajo (marcar mas cancelaciones) para evitar los costosos FN a
cambio de algunos FP baratos.

Este modulo barre todos los umbrales, calcula el coste total de negocio en cada
uno y encuentra el que lo minimiza, comparandolo con el 0.5 por defecto.

RIGOR
-----
La proporcion de costes (5:1) es una hipotesis de negocio PARAMETRIZABLE. Lo
relevante metodologicamente es: (1) que FN > FP de forma justificada, y (2) que
el umbral se elige sobre una funcion de coste explicita, no arbitrariamente.

Uso:
    python -m src.analisis_umbral
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from src.data_loader import SEED, preparar_datos

PATH_PROYECTO: Path = Path(__file__).resolve().parent.parent
PATH_FIGURAS: Path = PATH_PROYECTO / "resultados" / "figuras"
PATH_METRICAS: Path = PATH_PROYECTO / "resultados" / "metricas"

# --- Supuesto de coste de negocio (parametrizable) ---
COSTE_FN: float = 5.0
"""Coste de un falso negativo (cancelacion no anticipada -> noche perdida)."""

COSTE_FP: float = 1.0
"""Coste de un falso positivo (accion de retencion innecesaria)."""

# Hiperparametros del RF (mismos que el experimento de balanceo, para coherencia)
HIPERPARAMS_RF: dict = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": SEED,
}


def calcular_coste(
    y_true: pd.Series,
    y_proba: np.ndarray,
    umbral: float,
    coste_fn: float = COSTE_FN,
    coste_fp: float = COSTE_FP,
) -> dict[str, float]:
    """Calcula el coste de negocio total para un umbral dado.

    Args:
        y_true: Etiquetas reales.
        y_proba: Probabilidades predichas de la clase positiva.
        umbral: Umbral de decision (proba >= umbral -> clase 1).
        coste_fn: Coste unitario de un falso negativo.
        coste_fp: Coste unitario de un falso positivo.

    Returns:
        dict: Numero de FN, FP y coste total.
    """
    y_pred = (y_proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    coste_total = fn * coste_fn + fp * coste_fp
    return {"umbral": umbral, "fn": int(fn), "fp": int(fp), "coste_total": coste_total}


def barrer_umbrales(
    y_true: pd.Series,
    y_proba: np.ndarray,
    coste_fn: float = COSTE_FN,
    coste_fp: float = COSTE_FP,
    n_umbrales: int = 101,
) -> pd.DataFrame:
    """Calcula el coste para una rejilla de umbrales entre 0 y 1.

    Returns:
        pd.DataFrame: Una fila por umbral con fn, fp y coste_total.
    """
    umbrales = np.linspace(0.0, 1.0, n_umbrales)
    filas = [calcular_coste(y_true, y_proba, u, coste_fn, coste_fp) for u in umbrales]
    return pd.DataFrame(filas)


def ejecutar_analisis() -> tuple[pd.DataFrame, float, RandomForestClassifier]:
    """Entrena el RF, barre umbrales y encuentra el optimo por coste.

    Returns:
        tuple: (tabla_costes, umbral_optimo, modelo_entrenado).
    """
    print("Preparando datos...")
    X_train, X_test, y_train, y_test, _ = preparar_datos()

    print("Entrenando RandomForest...")
    modelo = RandomForestClassifier(**HIPERPARAMS_RF)
    modelo.fit(X_train, y_train)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    print(f"Barriendo umbrales (coste FN={COSTE_FN}, FP={COSTE_FP})...")
    tabla = barrer_umbrales(y_test, y_proba)

    idx_optimo = tabla["coste_total"].idxmin()
    umbral_optimo = float(tabla.loc[idx_optimo, "umbral"])

    return tabla, umbral_optimo, modelo


def imprimir_conclusion(tabla: pd.DataFrame, umbral_optimo: float) -> None:
    """Compara el umbral por defecto (0.5) con el optimo por coste."""
    fila_05 = tabla.iloc[(tabla["umbral"] - 0.5).abs().idxmin()]
    fila_opt = tabla.iloc[(tabla["umbral"] - umbral_optimo).abs().idxmin()]

    ahorro = fila_05["coste_total"] - fila_opt["coste_total"]
    ahorro_pct = (ahorro / fila_05["coste_total"] * 100) if fila_05["coste_total"] else 0

    print("\n" + "=" * 70)
    print("ANALISIS DE UMBRAL POR COSTE DE NEGOCIO")
    print(
        f"Supuesto: coste FN = {COSTE_FN}, coste FP = {COSTE_FP} (ratio {COSTE_FN / COSTE_FP:.0f}:1)"
    )
    print("=" * 70)
    print(f"\n{'Umbral':<12}{'FN':>8}{'FP':>8}{'Coste total':>15}")
    print("-" * 43)
    print(
        f"{'0.50 (def)':<12}{int(fila_05['fn']):>8}{int(fila_05['fp']):>8}{fila_05['coste_total']:>15.0f}"
    )
    print(
        f"{f'{umbral_optimo:.2f} (opt)':<12}{int(fila_opt['fn']):>8}{int(fila_opt['fp']):>8}{fila_opt['coste_total']:>15.0f}"
    )

    print("\n" + "=" * 70)
    print(f"UMBRAL OPTIMO: {umbral_optimo:.2f}")
    print(f"  Ahorro de coste vs umbral 0.5: {ahorro:.0f} ({ahorro_pct:.1f}%)")
    if umbral_optimo < 0.5:
        print("  El optimo es MENOR que 0.5: conviene marcar mas cancelaciones")
        print("  (asumir mas FP baratos para evitar FN caros).")
    elif umbral_optimo > 0.5:
        print("  El optimo es MAYOR que 0.5: conviene ser mas conservador.")
    print("=" * 70)


def graficar_coste(tabla: pd.DataFrame, umbral_optimo: float, guardar: bool = True) -> None:
    """Grafica la curva de coste total frente al umbral."""
    PATH_FIGURAS.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))
    plt.plot(
        tabla["umbral"],
        tabla["coste_total"],
        color="#2E75B6",
        linewidth=2,
        label="Coste total de negocio",
    )
    plt.axvline(0.5, color="grey", linestyle="--", alpha=0.7, label="Umbral por defecto (0.5)")
    plt.axvline(
        umbral_optimo,
        color="#C00000",
        linestyle="-",
        linewidth=2,
        label=f"Umbral optimo ({umbral_optimo:.2f})",
    )

    idx = (tabla["umbral"] - umbral_optimo).abs().idxmin()
    plt.scatter([umbral_optimo], [tabla.loc[idx, "coste_total"]], color="#C00000", zorder=5, s=60)

    plt.xlabel("Umbral de decision", fontsize=12)
    plt.ylabel(f"Coste total (FN x {COSTE_FN:.0f} + FP x {COSTE_FP:.0f})", fontsize=12)
    plt.title("Umbral optimo segun coste de negocio (RandomForest)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if guardar:
        ruta = PATH_FIGURAS / "umbral_coste.png"
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"\nFigura guardada: {ruta}")

    plt.show()


def main() -> None:
    """Ejecuta el analisis completo: tabla + optimo + grafica."""
    tabla, umbral_optimo, _modelo = ejecutar_analisis()
    imprimir_conclusion(tabla, umbral_optimo)
    graficar_coste(tabla, umbral_optimo)

    PATH_METRICAS.mkdir(parents=True, exist_ok=True)
    ruta_csv = PATH_METRICAS / "analisis_umbral.csv"
    tabla.round(4).to_csv(ruta_csv, index=False)
    print(f"Tabla guardada: {ruta_csv}")


if __name__ == "__main__":
    main()
