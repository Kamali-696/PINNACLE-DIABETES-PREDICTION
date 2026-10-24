import logging

from fastapi import APIRouter, Body, HTTPException, Request, status

from app.schemas import (
    EHRFeatures,
    ExplanationResponse,
    ModelPredictionResponse,
    PimaFeatures,
    UnifiedExplanationResponse,
    UnifiedPredictionResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prediction"])

# The existing React form and its saved feature schema identify the EHR pipeline
# as the primary prediction workflow. This fixed internal mapping avoids inferring
# a model from a person's health values or exposing model selection in the UI.
PRIMARY_MODEL_KEY = "ehr"


def _service(request: Request):
    return request.app.state.model_service


def _predict(dataset: str, payload: dict, request: Request) -> ModelPredictionResponse:
    try:
        result = _service(request).predict(dataset, payload)
        return ModelPredictionResponse(**result)
    except Exception as exc:
        logger.exception("Prediction failed for dataset %s.", dataset)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction could not be generated for this input.",
        ) from exc


def _explain(dataset: str, payload: dict, request: Request) -> ExplanationResponse:
    try:
        result = _service(request).explain(dataset, payload)
        return ExplanationResponse(**result)
    except Exception as exc:
        logger.exception("Explanation failed for dataset %s.", dataset)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Explanation could not be generated for this input.",
        ) from exc


@router.post("/predict/ehr", response_model=ModelPredictionResponse)
def predict_ehr(features: EHRFeatures, request: Request) -> ModelPredictionResponse:
    return _predict("ehr", features.model_dump(), request)


@router.post("/predict/pima", response_model=ModelPredictionResponse)
def predict_pima(features: PimaFeatures, request: Request) -> ModelPredictionResponse:
    return _predict("pima", features.model_dump(), request)


@router.post("/explain/ehr", response_model=ExplanationResponse)
def explain_ehr(features: EHRFeatures, request: Request) -> ExplanationResponse:
    return _explain("ehr", features.model_dump(), request)


@router.post("/explain/pima", response_model=ExplanationResponse)
def explain_pima(features: PimaFeatures, request: Request) -> ExplanationResponse:
    return _explain("pima", features.model_dump(), request)


@router.post("/predict", response_model=UnifiedPredictionResponse)
def predict_primary(features: EHRFeatures, request: Request) -> UnifiedPredictionResponse:
    """Unified public endpoint backed by the designated primary saved pipeline."""
    result = _predict(PRIMARY_MODEL_KEY, features.model_dump(), request)
    return UnifiedPredictionResponse(
        predicted_class=result.predicted_class,
        positive_class_probability=result.positive_class_probability,
    )


@router.post("/explain", response_model=UnifiedExplanationResponse)
def explain_primary(features: EHRFeatures, request: Request) -> UnifiedExplanationResponse:
    """Explain the same primary pipeline used by POST /predict."""
    result = _explain(PRIMARY_MODEL_KEY, features.model_dump(), request)
    return UnifiedExplanationResponse(
        predicted_class=result.predicted_class,
        positive_class_probability=result.positive_class_probability,
        output_scale=result.output_scale,
        baseline_raw_margin=result.baseline_raw_margin,
        final_raw_margin=result.final_raw_margin,
        reconstruction_error=result.reconstruction_error,
        top_contributions=result.top_contributions,
        interpretation_note=result.interpretation_note,
    )


@router.post("/predict/{dataset}")
def unsupported_prediction_dataset(dataset: str, request: Request, payload: dict = Body(...)) -> None:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Unsupported dataset. Choose the explicit EHR or Pima prediction endpoint.",
    )


@router.post("/explain/{dataset}")
def unsupported_explanation_dataset(dataset: str, request: Request, payload: dict = Body(...)) -> None:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Unsupported dataset. Choose the explicit EHR or Pima explanation endpoint.",
    )
