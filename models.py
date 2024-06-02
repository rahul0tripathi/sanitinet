from attrs import define
from typing import List
from pydantic import BaseModel


class RawResult(BaseModel):
    label: str
    score: float


class ClassificationResult(BaseModel):
    is_hate_speech: bool
    input: str
    raw: RawResult


class ClassificationRequest(BaseModel):
    input_text: str
