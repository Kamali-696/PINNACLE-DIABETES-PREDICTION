"""Request and response schemas for the independent EHR and Pima models."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator


class InputBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EHRFeatures(InputBase):
    """Feature names and categories match the saved EHR preprocessing pipeline."""

    gender: Literal["Female", "Male", "Other"]
    age: Annotated[StrictFloat, Field(ge=0, le=120, description="Age in years")]
    hypertension: Annotated[StrictInt, Field(ge=0, le=1)]
    heart_disease: Annotated[StrictInt, Field(ge=0, le=1)]
    smoking_history: Literal["never", "former", "current", "not current", "ever", "No Info"]
    bmi: Annotated[StrictFloat, Field(ge=10, le=100, description="BMI in kg/m^2")]
    HbA1c_level: Annotated[StrictFloat, Field(ge=3, le=20, description="HbA1c percentage")]
    blood_glucose_level: Annotated[StrictFloat, Field(ge=20, le=1000, description="Blood glucose in mg/dL")]


class PimaFeatures(InputBase):
    """Pima fields retain the exact case-sensitive names from the saved model."""

    Pregnancies: Annotated[StrictInt, Field(ge=0, le=30)]
    Glucose: Annotated[StrictFloat, Field(ge=0, le=400, description="Pima 2-hour OGTT glucose measurement")]
    BloodPressure: Annotated[StrictFloat, Field(ge=0, le=200, description="Diastolic blood pressure, mmHg")]
    SkinThickness: Annotated[StrictFloat, Field(ge=0, le=100, description="Triceps skin-fold thickness, mm")]
    Insulin: Annotated[StrictFloat, Field(ge=0, le=1000, description="Serum insulin measurement")]
    BMI: Annotated[StrictFloat, Field(ge=0, le=80, description="BMI in kg/m^2")]
    DiabetesPedigreeFunction: Annotated[StrictFloat, Field(ge=0, le=10)]
    Age: Annotated[StrictFloat, Field(ge=18, le=120, description="Age in years")]

    @model_validator(mode="after")
    def validate_preprocessing_ranges(self):
        # Zero is accepted as the dataset's missing-measurement sentinel for these
        # fields; nonzero values must still satisfy the preprocessing plausibility bounds.
        ranges = {
            "Glucose": (20, 400),
            "BloodPressure": (20, 200),
            "SkinThickness": (1, 100),
            "Insulin": (1, 1000),
            "BMI": (10, 80),
        }
        for name, (lower, upper) in ranges.items():
            value = getattr(self, name)
            if value != 0 and not lower <= value <= upper:
                raise ValueError(f"{name} must be 0 (missing sentinel) or within [{lower}, {upper}].")
        return self


class ModelPredictionResponse(BaseModel):
    dataset: Literal["ehr", "pima"]
    target: str
    model_name: str
    predicted_class: Literal[0, 1]
    positive_class_probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)


class FeatureContribution(BaseModel):
    feature: str
    input_value: str | float | int
    shap_value: float
    direction: Literal["toward_positive", "toward_negative", "no_effect"]


class ExplanationResponse(ModelPredictionResponse):
    output_scale: str
    baseline_raw_margin: float
    final_raw_margin: float
    reconstruction_error: float
    top_contributions: list[FeatureContribution]
    interpretation_note: str


class UnifiedPredictionResponse(BaseModel):
    """Public prediction shape for the fixed primary model route."""

    predicted_class: Literal[0, 1]
    positive_class_probability: float = Field(ge=0, le=1)


class UnifiedExplanationResponse(UnifiedPredictionResponse):
    output_scale: str
    baseline_raw_margin: float
    final_raw_margin: float
    reconstruction_error: float
    top_contributions: list[FeatureContribution]
    interpretation_note: str
