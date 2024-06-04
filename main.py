from text_processor import TextProcessor
from image_processor import ImageProcessor
from fastapi import FastAPI, UploadFile, HTTPException
from PIL import Image
import logging
import models

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
text_processor = TextProcessor()
image_processor = ImageProcessor()

@app.post("/v1/chat/classify", response_model=models.TextClassificationResult)
async def classify_text(input: models.TextClassificationRequest):
    try:
        is_hate_speech, raw = text_processor.is_hate_speech(input.input_text)
        return models.TextClassificationResult(
            is_hate_speech=is_hate_speech,
            input=input.input_text,
            raw=models.RawResult(label=raw['label'], score=raw['score'])
        )
    except Exception as e:
        logger.error(f"Error classifying text: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/v1/image/classify")
async def classify_image(input: UploadFile):
    try:
        image = Image.open(input.file)
        result = image_processor.classify(image=image)
        return result
    except Exception as e:
        logger.error(f"Error classifying image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)