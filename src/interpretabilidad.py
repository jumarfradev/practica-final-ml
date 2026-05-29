"""Interpretabilidad del modelo con SHAP (bonus de interpretabilidad avanzada).

Mientras que feature_importances_ dice que variables son importantes EN GENERAL,
SHAP explica POR QUE el modelo toma una decision concreta: cuanto empuja cada
variable, en cada prediccion individual, hacia "cancela" o "no cancela".

Dos niveles de explicacion:

1. GLOBAL (explicar_global): importancia media de cada variable sobre todo el
   conjunto, via valores SHAP. Complementa feature_importances_ con signo y
   magnitud por variable.

2. LOCAL (explicar_reserva): para UNA reserva concreta, las variables que mas
   empujaron la prediccion. Es lo que alimenta el endpoint /predict_explain de
   la API.

Funciona con TreeExplainer, exacto y rapido para modelos de arboles como el
RandomForest ganador.

Uso como modulo (lo usa la API):
    from src.interpretabilidad import crear_explainer, explicar_reserva
    explainer = crear_explainer(bundle)
    top = explicar_reserva(df_crudo, bundle, explainer)

Uso como script (genera la grafica SHAP global):
    python -m src.interpretabilidad
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.predictor import _preparar_datos_crudos, cargar_bundle

PATH_PROYECTO: Path = Path(__file__).resolve().parent.parent
PATH_FIGURAS: Path = PATH_PROYECTO / "resultados" / "figuras"


def crear_explainer(bundle: dict):
    """Crea un TreeExplainer de SHAP para el modelo del bundle.

    Debe crearse UNA vez y reutilizarse (es costoso). La API lo crea al arrancar.

    Args:
        bundle: Bundle cargado (debe contener un modelo de arboles).

    Returns:
        shap.TreeExplainer: Explainer listo para calcular valores SHAP.

    Raises:
        ValueError: Si el modelo no es de arboles (SHAP TreeExplainer no aplica).
    """
    import shap

    modelo = bundle["modelo"]
    if bundle.get("es_keras"):
        raise ValueError(
            "El modelo ganador es una red neuronal (Keras); TreeExplainer no "
            "aplica. SHAP para redes requiere otro explainer (DeepExplainer)."
        )
    return shap.TreeExplainer(modelo)


def _valores_shap_clase_positiva(explainer, X: np.ndarray) -> np.ndarray:
    """Extrae los valores SHAP de la clase positiva (cancela), robusto a versiones.

    Distintas versiones de SHAP devuelven el array de formas diferentes para
    clasificacion binaria (lista de 2 arrays, o array 3D). Esta funcion
    normaliza a un array 2D (filas x features) de la clase positiva.
    """
    valores = explainer.shap_values(X)

    # Caso 1: lista [clase_0, clase_1] (versiones antiguas)
    if isinstance(valores, list):
        return valores[1]

    # Caso 2: array 3D (n, features, clases) (versiones nuevas)
    if valores.ndim == 3:
        return valores[:, :, 1]

    # Caso 3: ya es 2D
    return valores


def explicar_reserva(
    df_crudo: pd.DataFrame,
    bundle: dict,
    explainer,
    top_n: int = 4,
) -> list[dict]:
    """Explica la prediccion de UNA reserva: las top_n variables mas influyentes.

    Args:
        df_crudo: DataFrame con UNA reserva en formato crudo.
        bundle: Bundle cargado.
        explainer: TreeExplainer ya creado con crear_explainer().
        top_n: Numero de variables a devolver. Defaults to 4.

    Returns:
        list[dict]: Lista de top_n variables ordenadas por influencia absoluta,
            cada una con su nombre, el valor SHAP y la direccion ('aumenta' o
            'reduce' la probabilidad de cancelacion).
    """
    X = _preparar_datos_crudos(df_crudo, bundle)
    nombres = bundle["nombres_features"]

    shap_vals = _valores_shap_clase_positiva(explainer, X)
    fila = shap_vals[0]  # primera (y unica) reserva

    # Ordenar por magnitud absoluta
    indices = np.argsort(np.abs(fila))[::-1][:top_n]

    explicacion = []
    for idx in indices:
        valor = float(fila[idx])
        explicacion.append(
            {
                "variable": nombres[idx],
                "impacto_shap": round(valor, 4),
                "direccion": "aumenta" if valor > 0 else "reduce",
            }
        )
    return explicacion


def explicar_global(bundle: dict, X_muestra: np.ndarray, guardar: bool = True) -> None:
    """Genera una grafica SHAP global (summary plot) sobre una muestra.

    Args:
        bundle: Bundle cargado.
        X_muestra: Matriz de features ya preprocesada (muestra del conjunto).
        guardar: Si True, guarda la figura en resultados/figuras/.
    """
    import matplotlib.pyplot as plt
    import shap

    explainer = crear_explainer(bundle)
    shap_vals = _valores_shap_clase_positiva(explainer, X_muestra)

    PATH_FIGURAS.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(
        shap_vals,
        X_muestra,
        feature_names=bundle["nombres_features"],
        show=False,
        max_display=15,
    )
    if guardar:
        ruta = PATH_FIGURAS / "shap_summary.png"
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"Figura SHAP global guardada: {ruta}")
    plt.close()


def main() -> None:
    """Genera la grafica SHAP global usando el modelo persistido."""
    print("Cargando modelo...")
    bundle = cargar_bundle()

    if bundle.get("es_keras"):
        print("El modelo ganador es Keras; SHAP TreeExplainer no aplica.")
        return

    # Usar una muestra del propio test para el summary (cargando datos)
    from src.data_loader import preparar_datos

    print("Preparando muestra de datos...")
    _X_train, X_test, _y_train, _y_test, _ = preparar_datos()

    # Muestra para que SHAP no tarde demasiado
    n = min(500, X_test.shape[0])
    X_muestra = X_test[:n]

    print(f"Calculando valores SHAP sobre {n} muestras...")
    explicar_global(bundle, X_muestra)
    print("OK")


if __name__ == "__main__":
    main()
