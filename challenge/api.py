from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List
import pandas as pd
from challenge.model import DelayModel

# Configure FastAPI application
app = FastAPI(
    title="Delay Prediction API",
    version="1.0",
    description="API for predicting flight delays using DelayModel.",
)

# Custom exception handler to return 400 on validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors(), "body": exc.body},
    )


# Pydantic model for a single flight input (only required fields)
class FlightData(BaseModel):
    OPERA: str = Field(..., example="Aerolineas Argentinas")
    TIPOVUELO: str = Field(..., example="N")
    MES: int = Field(..., example=3)

    class Config:
        extra = "forbid"  # Forbid any extra fields not defined here

    @validator("TIPOVUELO")
    # pylint: disable=no-self-argument
    def validate_tipovuelo(cls, v):
        if v not in ["I", "N"]:
            raise ValueError('TIPOVUELO must be "I" or "N"')
        return v

    @validator("MES")
    # pylint: disable=no-self-argument
    def validate_mes(cls, v):
        if not (1 <= v <= 12):
            raise ValueError("MES must be between 1 and 12.")
        return v


# Batch model for multiple flights
class FlightBatch(BaseModel):
    flights: List[FlightData]


# Initialize the DelayModel
try:
    model = DelayModel()
except Exception as e:
    raise RuntimeError(f"Failed to initialize DelayModel: {e}") from e

@app.get("/health", status_code=200)
async def get_health() -> dict:
    """
    Health check endpoint.

    Returns:
        dict: {"status": "OK"}
    """
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(batch: FlightBatch) -> dict:
    """
    Predict flight delays for a batch of flights.

    This endpoint receives a JSON payload with a list of flights, validates the input,
    converts it into a DataFrame, preprocesses it using the DelayModel, and returns predictions.

    Args:
        batch (FlightBatch): A JSON object containing flight data.

    Returns:
        dict: {"predict": [predictions]}

    Raises:
        HTTPException: If prediction fails.
    """
    try:
        # Convert the batch of flights to a DataFrame
        df = pd.DataFrame([flight.dict() for flight in batch.flights])
        # Preprocess the data using DelayModel (only using OPERA, TIPOVUELO, and MES)
        features = model.preprocess(df)
        predictions = model.predict(features)
        return {"predict": predictions}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
