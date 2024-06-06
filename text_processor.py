import logging
import time
import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline, BitsAndBytesConfig
)
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10
from nltk import sent_tokenize
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA
from IndicTransTokenizer import IndicProcessor
from langid.langid import LanguageIdentifier, model as langmodel
from load_models import get_hf_model, MODEL_INIDIC_TRANS, MODEL_HINGLISH_CLASSIFIER, MODEL_HATE_SPEECH_CLASSIFIER
from sacremoses import MosesTokenizer

# Mapping of language codes
flores_codes = {
    "asm_Beng": "as",
    "awa_Deva": "hi",
    "ben_Beng": "bn",
    "bho_Deva": "hi",
    "brx_Deva": "hi",
    "doi_Deva": "hi",
    "eng_Latn": "en",
    "gom_Deva": "kK",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "hne_Deva": "hi",
    "kan_Knda": "kn",
    "kas_Arab": "ur",
    "kas_Deva": "hi",
    "kha_Latn": "en",
    "lus_Latn": "en",
    "mag_Deva": "hi",
    "mai_Deva": "hi",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Beng": "bn",
    "mni_Mtei": "hi",
    "npi_Deva": "ne",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "san_Deva": "hi",
    "sat_Olck": "or",
    "snd_Arab": "ur",
    "snd_Deva": "hi",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}

logger = logging.getLogger(__name__)


class TextProcessor:
    model_0_tokenizer = None
    model_0 = None
    model_1_tokenizer = None
    model_1 = None
    model_1_pipeline = None
    model_2_tokenizer = None
    model_2 = None
    model_2_pipeline = None
    indic_preprocessor = None
    lang_classifier = None
    moses_splitter_en = None
    attn_implementation = "eager"
    quantization = ""
    BATCH_SIZE = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self) -> None:
        logger.debug("Initializing processor")
        self.initialize_indic_model_and_tokenizer()
        self.initialize_hinglish_model_and_tokenizer()
        self.initialize_hate_speech_model_and_tokenizer()
        self.initialize_lang_classifier()
        self.initialize_moses_splitter_en()

    def initialize_indic_model_and_tokenizer(self):
        logger.debug("Initializing Indic model and tokenizer")
        self.indic_preprocessor = IndicProcessor(inference=True)

        # Configure quantization
        if self.quantization == "4-bit":
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self.quantization == "8-bit":
            qconfig = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        else:
            qconfig = None

        # Configure attention implementation
        if self.attn_implementation == "flash_attention_2":
            if is_flash_attn_2_available() and is_flash_attn_greater_or_equal_2_10():
                self.attn_implementation = "flash_attention_2"
            else:
                self.attn_implementation = "eager"

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            get_hf_model(MODEL_INIDIC_TRANS), trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            get_hf_model(MODEL_INIDIC_TRANS),
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            low_cpu_mem_usage=True,
            quantization_config=qconfig,
        )

        if qconfig is None:
            model = model.to(self.DEVICE)
            model.half()

        model.eval()

        self.model_0 = model
        self.model_0_tokenizer = tokenizer

    def initialize_hinglish_model_and_tokenizer(self):
        logger.debug("Initializing Hinglish model and tokenizer")
        self.model_1_tokenizer = AutoTokenizer.from_pretrained(
            get_hf_model(MODEL_HINGLISH_CLASSIFIER))
        self.model_1 = AutoModelForSequenceClassification.from_pretrained(
            get_hf_model(MODEL_HINGLISH_CLASSIFIER)).to(self.DEVICE)
        self.model_1_pipeline = pipeline(
            "text-classification", model=self.model_1, tokenizer=self.model_1_tokenizer, device=0 if self.DEVICE == "cuda" else -1)

    def initialize_hate_speech_model_and_tokenizer(self):
        logger.debug("Initializing hate speech model and tokenizer")
        self.model_2_tokenizer = AutoTokenizer.from_pretrained(
            get_hf_model(MODEL_HATE_SPEECH_CLASSIFIER))
        self.model_2 = AutoModelForSequenceClassification.from_pretrained(
            get_hf_model(MODEL_HATE_SPEECH_CLASSIFIER)).to(self.DEVICE)
        self.model_2 = self.model_2.to_bettertransformer()
        self.model_2_pipeline = pipeline(
            "text-classification", model=self.model_2, tokenizer=self.model_2_tokenizer, device=0 if self.DEVICE == "cuda" else -1)

    def initialize_lang_classifier(self):
        logger.debug("Initializing language classifier")
        self.lang_classifier = LanguageIdentifier.from_modelstring(
            langmodel, norm_probs=True)

    def initialize_moses_splitter_en(self):
        logger.debug("Initializing Moses sentence splitter for English")
        self.moses_splitter_en = MosesTokenizer(
            flores_codes["eng_Latn"])

    def split_sentences(self, input_text, lang):
        logger.debug(f"Splitting sentences for language: {lang}")
        if lang == "eng_Latn":
            sents_moses = self.moses_splitter_en.tokenize([input_text])
            sents_nltk = sent_tokenize(input_text)
            input_sentences = sents_nltk if len(
                sents_nltk) < len(sents_moses) else sents_moses
            input_sentences = [sent.replace("\xad", "")
                               for sent in input_sentences]
        else:
            input_sentences = sentence_split(
                input_text, lang=flores_codes[lang], delim_pat=DELIM_PAT_NO_DANDA)
        return input_sentences

    def batch_translate(self, input_sentences, src_lang, tgt_lang):
        logger.debug(f"Batch translating from {src_lang} to {tgt_lang}")
        translations = []
        for i in range(0, len(input_sentences), self.BATCH_SIZE):
            batch = input_sentences[i: i + self.BATCH_SIZE]

            # Preprocess the batch and extract entity mappings
            batch = self.indic_preprocessor.preprocess_batch(
                batch, src_lang=src_lang, tgt_lang=tgt_lang)

            # Tokenize the batch and generate input encodings
            inputs = self.model_0_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.DEVICE)

            # Generate translations using the model
            with torch.no_grad():
                generated_tokens = self.model_0.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            # Decode the generated tokens into text
            with self.model_0_tokenizer.as_target_tokenizer():
                generated_tokens = self.model_0_tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            # Postprocess the translations, including entity replacement
            translations += self.indic_preprocessor.postprocess_batch(
                generated_tokens, lang=tgt_lang)

            del inputs
            torch.cuda.empty_cache()

        return translations

    def translate_paragraph(self, input_text, src_lang, tgt_lang):
        logger.debug(f"Translating paragraph from {src_lang} to {tgt_lang}")
        input_sentences = self.split_sentences(input_text, src_lang)
        translated_text = self.batch_translate(
            input_sentences, src_lang, tgt_lang)
        return " ".join(translated_text)

    def process_input(self, input_text):
        logger.debug(
            "Processing input text for Hinglish classification and translation")
        hinglish_classification = self.model_1_pipeline(input_text)
        logger.debug(
            f"Hinglish classification result: {hinglish_classification}")

        if hinglish_classification[0]['label'] in ['LABEL_1', 'LABEL_0']:
            lang = self.lang_classifier.classify(input_text)
            logger.debug(f"Language classification result: {lang}")

            if lang[0] in ['as', 'hi', 'bn', 'kK', 'gu', 'kn', 'ur', 'ml', 'mr', 'ne', 'or', 'pa', 'ta', 'te']:
                return input_text

            src_lang, tgt_lang = "eng_Latn", "hin_Deva"
            hi_translations = self.translate_paragraph(
                input_text, src_lang, tgt_lang)
            logger.debug(f"Translated text to Hindi: {hi_translations}")
            return hi_translations
        else:
            return input_text

    def is_hate_speech(self, input_text):
        logger.debug("Processing text for hate speech classification")
        input_processed = self.process_input(input_text)
        logger.debug(f"Processed input text: {input_processed}")
        output = self.model_2_pipeline(input_processed)
        logger.debug(f"Hate speech classification result: {output}")
        return output[0]['label'] == 'LABEL_1', output[0]
