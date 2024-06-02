from processor import Processor
import json

from fastapi import FastAPI, HTTPException
from typing import List
import logging
import models

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = FastAPI()
processor = Processor()


@app.post("/v1/classify", response_model=models.ClassificationResult)
async def classify(input: models.ClassificationRequest):
    is_hate_speech, raw = processor.is_hate_speech(
        input.input_text)
    return models.ClassificationResult(is_hate_speech=is_hate_speech, input=input.input_text, raw=models.RawResult(label=raw['label'], score=raw['score']))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
