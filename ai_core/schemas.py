from pydantic import BaseModel, Field

class EpidemicPrediction(BaseModel):
    event_id: int = Field(..., description="Unique identification number for the epidemic event")
    disease: str = Field(..., description="Name of the disease or outbreak pathogen")
    region: str = Field(..., description="Geographic region/location of the outbreak")
    confidence_score: float = Field(..., description="Confidence score of the prediction, ranging from 0.0 to 1.0")
    explainable_reasoning: str = Field(..., description="Explainable reasoning and clinical/epidemiological justification for the prediction")
