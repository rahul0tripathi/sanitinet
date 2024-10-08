import logging
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    pipeline
)
from load_models import get_hf_model, MODEL_NSFW_IMAGE_CLASSIFIER
from models import ImageClassificationResult
import torch

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self) -> None:
        logger.debug("Initializing image processor")
        model_name = get_hf_model(MODEL_NSFW_IMAGE_CLASSIFIER)
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name).to(DEVICE)
        self.pipeline = pipeline(
            "image-classification", model=self.model, image_processor=self.image_processor, device=0 if DEVICE == "cuda" else -1
        )

    def classify(self, image) -> ImageClassificationResult:
        output = self.pipeline(image)
        nsfw = any(
            l['label'] in ['hentai', 'sexy', 'porn'] and l['score'] > 0.5
            for l in output
        )
        return ImageClassificationResult(nsfw=nsfw, raw=output)
