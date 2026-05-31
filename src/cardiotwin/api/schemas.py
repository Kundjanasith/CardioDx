from pydantic import BaseModel, Field
from typing import List, Optional

class ECGUpload(BaseModel):
    fs: float = Field(default=500.0)
    leads: List[str]
    samples: List[List[float]]
    record_id: str = "api_upload"

class PredictionResponse(BaseModel):
    record_id: str
    state: dict
