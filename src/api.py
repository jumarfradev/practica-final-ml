"""API REST de prediccion de cancelaciones hoteleras (bonus FastAPI).

Expone el modelo entrenado (models/best_model.pkl) como un servicio web. Recibe
reservas en formato JSON y devuelve la probabilidad de cancelacion, reutilizando
la cadena de inferencia de src.predictor (datos crudos -> limpieza -> top-N
congelado -> preprocesador -> modelo).

Endpoints:
    GET  /health         -> estado del servicio y si el modelo esta cargado.
    GET  /model_info     -> metadatos del modelo ganador (nombre, metricas, fecha).
    POST /predict        -> prediccion para UNA reserva.
    POST /predict_batch  -> prediccion para VARIAS reservas.

Ejecutar en local:
    uvicorn src.api:app --reload

Documentacion interactiva (Swagger) disponible en:
    http://127.0.0.1:8000/docs

Diseno de campos (opcion 1B): se exigen obligatorios solo los campos mas
influyentes; el resto tienen valores por defecto razonables para facilitar las
pruebas y la demo. En produccion real se exigirian todos.
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predictor import cargar_bundle, predecir

# ============================================================================
# Estado global: el bundle se carga una vez al arrancar
# ============================================================================

_estado: dict = {"bundle": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar la API y lo libera al cerrar.

    Cargar el bundle una sola vez (no en cada peticion) es clave para el
    rendimiento: el .pkl con el RandomForest puede ser grande.
    """
    try:
        _estado["bundle"] = cargar_bundle()
        print(f"OK - Modelo cargado: {_estado['bundle']['nombre_modelo']}")
    except FileNotFoundError as e:
        # La API arranca igualmente; /health avisara de que no hay modelo.
        print(f"WARN - No se pudo cargar el modelo: {e}")
        _estado["bundle"] = None
    yield
    _estado["bundle"] = None


app = FastAPI(
    title="API de Prediccion de Cancelaciones Hoteleras",
    description=(
        "Predice la probabilidad de que una reserva de hotel sea cancelada. "
        "Modelo ganador entrenado en la practica final de ML/DL (PontIA.tech)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Esquemas de entrada/salida (Pydantic v2)
# ============================================================================


class Reserva(BaseModel):
    """Reserva hotelera en formato crudo para predecir su cancelacion.

    Campos OBLIGATORIOS: los mas influyentes segun feature_importances_.
    Campos OPCIONALES: tienen valores por defecto razonables (medianas/modas
    aproximadas del dataset) para que la demo no requiera rellenar 30 campos.
    """

    # --- Obligatorios (los mas influyentes) ---
    lead_time: int = Field(..., ge=0, description="Dias entre reserva y llegada.")
    hotel: str = Field(..., description="'City Hotel' o 'Resort Hotel'.")
    deposit_type: str = Field(..., description="'No Deposit', 'Non Refund' o 'Refundable'.")
    country: str = Field(..., description="Codigo de pais ISO (ej. 'PRT', 'ESP').")
    adr: float = Field(..., description="Average Daily Rate (precio medio/noche).")
    market_segment: str = Field(
        ..., description="Canal: 'Online TA', 'Offline TA/TO', 'Direct', 'Groups'..."
    )
    total_of_special_requests: int = Field(
        ..., ge=0, description="Numero de peticiones especiales."
    )

    # --- Opcionales con defaults razonables ---
    arrival_date_year: int = 2016
    arrival_date_month: str = "August"
    arrival_date_week_number: int = 27
    arrival_date_day_of_month: int = 15
    stays_in_weekend_nights: int = 1
    stays_in_week_nights: int = 2
    adults: int = 2
    children: int = 0
    babies: int = 0
    meal: str = "BB"
    distribution_channel: str = "TA/TO"
    is_repeated_guest: int = 0
    previous_cancellations: int = 0
    previous_bookings_not_canceled: int = 0
    reserved_room_type: str = "A"
    assigned_room_type: str = "A"
    booking_changes: int = 0
    agent: float | None = None
    company: float | None = None
    days_in_waiting_list: int = 0
    customer_type: str = "Transient"
    required_car_parking_spaces: int = 0

    model_config = {
        "json_schema_extra": {
            "example": {
                "lead_time": 120,
                "hotel": "City Hotel",
                "deposit_type": "No Deposit",
                "country": "PRT",
                "adr": 95.0,
                "market_segment": "Online TA",
                "total_of_special_requests": 0,
            }
        }
    }


class RespuestaPrediccion(BaseModel):
    """Respuesta de una prediccion individual."""

    probabilidad_cancelacion: float = Field(
        ..., description="Probabilidad estimada de cancelacion (0 a 1)."
    )
    prediccion: int = Field(..., description="0 = no cancela, 1 = cancela.")
    etiqueta: str = Field(..., description="Texto legible de la prediccion.")


class PeticionBatch(BaseModel):
    """Peticion con varias reservas."""

    reservas: list[Reserva]


# ============================================================================
# Utilidad interna
# ============================================================================


def _obtener_bundle() -> dict:
    """Devuelve el bundle cargado o lanza 503 si no esta disponible."""
    bundle = _estado.get("bundle")
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta 'python trainer.py' y reinicia la API.",
        )
    return bundle


def _a_dataframe(reservas: list[Reserva]) -> pd.DataFrame:
    """Convierte una lista de reservas Pydantic en DataFrame crudo."""
    return pd.DataFrame([r.model_dump() for r in reservas])


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health", tags=["estado"])
def health() -> dict:
    """Comprueba que el servicio esta vivo y si el modelo esta cargado."""
    bundle = _estado.get("bundle")
    return {
        "status": "ok",
        "modelo_cargado": bundle is not None,
        "nombre_modelo": bundle["nombre_modelo"] if bundle else None,
    }


@app.get("/model_info", tags=["estado"])
def model_info() -> dict:
    """Devuelve los metadatos del modelo ganador en produccion."""
    bundle = _obtener_bundle()
    return {
        "nombre_modelo": bundle["nombre_modelo"],
        "metrica_principal": bundle.get("metrica_principal"),
        "metricas": bundle.get("metricas"),
        "n_features": len(bundle.get("nombres_features", [])),
        "fecha_entrenamiento": bundle.get("fecha_entrenamiento"),
    }


@app.post("/predict", response_model=RespuestaPrediccion, tags=["prediccion"])
def predict(reserva: Reserva) -> RespuestaPrediccion:
    """Predice la probabilidad de cancelacion de UNA reserva."""
    bundle = _obtener_bundle()
    df = _a_dataframe([reserva])

    try:
        resultado = predecir(df, bundle)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {e}") from e

    proba = float(resultado["probabilidad_cancelacion"].iloc[0])
    pred = int(resultado["prediccion"].iloc[0])

    return RespuestaPrediccion(
        probabilidad_cancelacion=round(proba, 4),
        prediccion=pred,
        etiqueta="Cancela" if pred == 1 else "No cancela",
    )


@app.post("/predict_batch", tags=["prediccion"])
def predict_batch(peticion: PeticionBatch) -> dict:
    """Predice la probabilidad de cancelacion de VARIAS reservas a la vez."""
    bundle = _obtener_bundle()

    if not peticion.reservas:
        raise HTTPException(status_code=400, detail="La lista de reservas esta vacia.")

    df = _a_dataframe(peticion.reservas)

    try:
        resultado = predecir(df, bundle)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {e}") from e

    predicciones = [
        {
            "probabilidad_cancelacion": round(float(p), 4),
            "prediccion": int(c),
            "etiqueta": "Cancela" if int(c) == 1 else "No cancela",
        }
        for p, c in zip(
            resultado["probabilidad_cancelacion"],
            resultado["prediccion"],
            strict=True,
        )
    ]

    return {"n_reservas": len(predicciones), "predicciones": predicciones}
