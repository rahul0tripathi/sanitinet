from attrs import define
from typing import List
from pydantic import BaseModel


class RawResult(BaseModel):
    label: str
    score: float


class TextClassificationResult(BaseModel):
    is_hate_speech: bool
    input: str
    raw: RawResult


class TextClassificationRequest(BaseModel):
    input_text: str


class ImageClassificationResult(BaseModel):
    nsfw: bool
    raw: List[RawResult]
