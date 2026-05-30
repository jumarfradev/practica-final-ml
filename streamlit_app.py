"""Interfaz visual (Streamlit) para la API de prediccion de cancelaciones.

Esta app NO carga el modelo directamente: se comunica con la API FastAPI por
HTTP (endpoints /predict y /predict_batch). Es decir, es un cliente de la API.
Demuestra el bonus de "interfaz visual" reutilizando el bonus de "API REST",
compartiendo un unico backend sin duplicar la logica de prediccion.

REQUISITO: la API debe estar arrancada antes de usar esta interfaz:
    Terminal 1:  uvicorn src.api:app
    Terminal 2:  streamlit run streamlit_app.py

Por defecto la API se espera en http://127.0.0.1:8000 (configurable abajo).
"""

import io

import pandas as pd
import requests
import streamlit as st

# ============================================================================
# Configuracion
# ============================================================================

API_URL = "http://127.0.0.1:8000"
TIMEOUT = 10  # segundos

st.set_page_config(
    page_title="Prediccion de Cancelaciones Hoteleras",
    page_icon=None,
    layout="centered",
)


# ============================================================================
# Utilidades de comunicacion con la API
# ============================================================================


def api_disponible() -> tuple[bool, dict | None]:
    """Comprueba si la API esta levantada y el modelo cargado.

    Returns:
        tuple: (esta_ok, info_health). esta_ok es True si la API responde y
            tiene el modelo cargado. info_health es el JSON de /health o None.
    """
    try:
        r = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            return data.get("modelo_cargado", False), data
        return False, None
    except requests.exceptions.RequestException:
        return False, None


def obtener_model_info() -> dict | None:
    """Pide los metadatos del modelo a /model_info."""
    try:
        r = requests.get(f"{API_URL}/model_info", timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        pass
    return None


def predecir_una(reserva: dict) -> dict | None:
    """Envia una reserva a /predict y devuelve la respuesta."""
    try:
        r = requests.post(f"{API_URL}/predict", json=reserva, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
        st.error(f"La API devolvio un error {r.status_code}: {r.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo contactar con la API: {e}")
    return None


def predecir_todos_modelos(reserva: dict) -> dict | None:
    """Envia una reserva a /predict_all (comparativa de todos los modelos)."""
    try:
        r = requests.post(f"{API_URL}/predict_all", json=reserva, timeout=TIMEOUT * 2)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 503:
            st.info(
                "La comparativa entre modelos no esta disponible. Requiere haber "
                "entrenado en modo completo (python trainer.py) para generar "
                "all_models.pkl."
            )
        else:
            st.error(f"La API devolvio un error {r.status_code}: {r.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo contactar con la API: {e}")
    return None


def predecir_lote(reservas: list[dict]) -> dict | None:
    """Envia varias reservas a /predict_batch y devuelve la respuesta."""
    try:
        r = requests.post(
            f"{API_URL}/predict_batch",
            json={"reservas": reservas},
            timeout=TIMEOUT * 3,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"La API devolvio un error {r.status_code}: {r.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo contactar con la API: {e}")
    return None


def mostrar_resultado(resultado: dict) -> None:
    """Muestra el resultado de una prediccion de forma visual."""
    proba = resultado["probabilidad_cancelacion"]
    etiqueta = resultado["etiqueta"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilidad de cancelacion", f"{proba * 100:.1f}%")
    with col2:
        if resultado["prediccion"] == 1:
            st.error(f"Prediccion: {etiqueta}")
        else:
            st.success(f"Prediccion: {etiqueta}")

    st.progress(min(max(proba, 0.0), 1.0))


# ============================================================================
# Cabecera y estado de la API
# ============================================================================

st.title("Prediccion de Cancelaciones Hoteleras")
st.caption(
    "Interfaz visual que consume la API FastAPI del proyecto. Practica final ML/DL - PontIA.tech"
)

ok, health = api_disponible()

if not ok:
    st.warning(
        f"No se detecta la API en {API_URL} (o el modelo no esta cargado).\n\n"
        "Arrancala en otra terminal con:\n\n"
        "    uvicorn src.api:app\n\n"
        "y asegurate de haber entrenado antes el modelo con 'python trainer.py'."
    )
    st.stop()

info = obtener_model_info()
if info:
    with st.expander("Informacion del modelo en produccion"):
        st.write(f"**Modelo:** {info.get('nombre_modelo')}")
        metricas = info.get("metricas", {})
        st.write(f"**ROC-AUC:** {metricas.get('roc_auc', 0):.4f}")
        st.write(f"**F1:** {metricas.get('f1', 0):.4f}")
        st.write(f"**Features:** {info.get('n_features')}")
        st.write(f"**Entrenado:** {info.get('fecha_entrenamiento')}")


# ============================================================================
# Pestanas: individual y lote
# ============================================================================

tab_individual, tab_lote = st.tabs(["Reserva individual", "Lote (CSV)"])

# --- Pestana 1: prediccion individual ---
with tab_individual:
    st.subheader("Datos de la reserva")
    st.caption("Solo los campos mas influyentes; el resto usan valores por defecto.")

    col1, col2 = st.columns(2)
    with col1:
        lead_time = st.number_input("Lead time (dias hasta llegada)", min_value=0, value=120)
        hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
        deposit_type = st.selectbox("Tipo de deposito", ["No Deposit", "Non Refund", "Refundable"])
        adr = st.number_input("ADR (precio medio/noche)", min_value=0.0, value=95.0)
    with col2:
        country = st.text_input("Pais (codigo ISO)", value="PRT")
        market_segment = st.selectbox(
            "Segmento de mercado",
            ["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate", "Complementary"],
        )
        total_of_special_requests = st.number_input("Peticiones especiales", min_value=0, value=0)

    if st.button("Predecir", type="primary"):
        reserva = {
            "lead_time": int(lead_time),
            "hotel": hotel,
            "deposit_type": deposit_type,
            "country": country.strip().upper(),
            "adr": float(adr),
            "market_segment": market_segment,
            "total_of_special_requests": int(total_of_special_requests),
        }
        resultado = predecir_una(reserva)
        if resultado:
            st.divider()
            mostrar_resultado(resultado)

            # --- Comparativa: como predecirian los demas modelos ---
            st.divider()
            st.subheader("Comparativa entre modelos")
            st.caption(
                "La prediccion oficial la realiza el modelo ganador. A continuacion, "
                "con fines demostrativos, se muestra como predeciria cada modelo esta "
                "misma reserva (en produccion solo se sirve el ganador)."
            )
            comparativa = predecir_todos_modelos(reserva)
            if comparativa and comparativa.get("predicciones"):
                import pandas as _pd

                df_comp = _pd.DataFrame(comparativa["predicciones"])
                df_comp["Probabilidad (%)"] = (df_comp["probabilidad_cancelacion"] * 100).round(1)
                df_comp["Prediccion"] = df_comp["prediccion"].map({1: "Cancela", 0: "No cancela"})
                df_comp = df_comp.rename(
                    columns={"modelo": "Modelo", "roc_auc_global": "ROC-AUC global"}
                )

                # Grafica de barras: probabilidad por modelo
                grafico = df_comp.set_index("Modelo")["Probabilidad (%)"]
                st.bar_chart(grafico)

                # Tabla comparativa
                st.dataframe(
                    df_comp[["Modelo", "Probabilidad (%)", "Prediccion", "ROC-AUC global"]],
                    use_container_width=True,
                    hide_index=True,
                )

# --- Pestana 2: prediccion por lote (CSV) ---
with tab_lote:
    st.subheader("Prediccion por lote")
    st.caption(
        "Sube un CSV con reservas en formato crudo (mismas columnas que el "
        "dataset original). Se enviaran a la API en una sola peticion."
    )

    archivo = st.file_uploader("Archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            df = pd.read_csv(archivo)
            st.write(f"Filas cargadas: {len(df)}")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Predecir lote", type="primary"):
                # Limitar a un numero razonable para la demo
                if len(df) > 500:
                    st.info("Se predeciran solo las primeras 500 filas (demo).")
                    df = df.head(500)

                reservas = df.to_dict(orient="records")
                respuesta = predecir_lote(reservas)

                if respuesta:
                    preds = pd.DataFrame(respuesta["predicciones"])
                    salida = pd.concat([df.reset_index(drop=True), preds], axis=1)
                    st.success(f"Predichas {respuesta['n_reservas']} reservas.")
                    st.dataframe(salida, use_container_width=True)

                    # Descarga del resultado
                    buffer = io.StringIO()
                    salida.to_csv(buffer, index=False)
                    st.download_button(
                        "Descargar resultados (CSV)",
                        data=buffer.getvalue(),
                        file_name="predicciones.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
