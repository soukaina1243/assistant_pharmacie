"""Schémas Pydantic de l'API (identiques au schéma JSON du rapport)."""
from typing import Any, Optional

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, examples=["wach kayn doliprane 1000 f rabat ?"])
    fulfill: bool = Field(True, description="Interroger les bases (prix, stock, pharmacies) en plus de l'analyse")


class InputInfo(BaseModel):
    modality: str
    raw_text: str
    transcript: Optional[str] = None


class LanguageInfo(BaseModel):
    primary: str
    script: str
    code_mixed: bool


class IntentInfo(BaseModel):
    label: str
    confidence: float


class Medication(BaseModel):
    raw: str
    name: str
    dosage: Optional[str] = None
    form: Optional[str] = None
    match_score: float
    candidates: list[str] = []


class Entities(BaseModel):
    medications: list[Medication] = []
    quantity: Optional[int] = None
    pharmacy_name: Optional[str] = None
    city: Optional[str] = None
    neighborhood: Optional[str] = None
    time_reference: Optional[str] = None


class AnalysisResponse(BaseModel):
    request_id: str
    input: InputInfo
    language: LanguageInfo
    intent: IntentInfo
    entities: Entities
    status: str = Field(..., description="ok | needs_clarification | out_of_scope")
    clarification_question: Optional[str] = None
    warnings: list[str] = []
    reply: Optional[str] = Field(None, description="Réponse en langage naturel (langue de l'utilisateur)")
    fulfillment: Optional[dict[str, Any]] = None
    meta: dict[str, Any] = {}


class BoxMatch(BaseModel):
    rank: int
    code: str
    name: str
    brand: str
    dci: Optional[str] = None
    dosage: Optional[str] = None
    dosage_mg: Optional[float] = None
    form: Optional[str] = None
    presentation: Optional[str] = None
    ppv_mad: Optional[float] = None
    reimbursement_pct: Optional[float] = None
    score: float
    evidence: list[str] = []


class BoxRecognitionResponse(BaseModel):
    request_id: str
    status: str = Field(..., description="ok | low_confidence | unknown | no_text")
    confidence: float
    matches: list[BoxMatch] = []
    detected: dict[str, Any] = {}
    ocr: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    warnings: list[str] = []
    advice: list[str] = []
    reply: Optional[str] = None
