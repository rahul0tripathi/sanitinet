import time
import torch
import torch.nn as nn

from mosestokenizer import MosesSentenceSplitter

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline, QuantoConfig, BitsAndBytesConfig
)
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10
from nltk import sent_tokenize
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA

from IndicTransTokenizer import IndicProcessor
from langid.langid import LanguageIdentifier, model as langmodel
quantization_config = QuantoConfig(weights="int2")

# FLORES language code mapping to 2 letter ISO language code for compatibility
# with Indic NLP Library (https://github.com/anoopkunchukuttan/indic_nlp_library)
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


def split_sentences(input_text, lang):
    if lang == "eng_Latn":
        input_sentences = sent_tokenize(input_text)
        with MosesSentenceSplitter(flores_codes[lang]) as splitter:
            sents_moses = splitter([input_text])
        sents_nltk = sent_tokenize(input_text)
        if len(sents_nltk) < len(sents_moses):
            input_sentences = sents_nltk
        else:
            input_sentences = sents_moses
        input_sentences = [sent.replace("\xad", "")
                           for sent in input_sentences]
    else:
        input_sentences = sentence_split(
            input_text, lang=flores_codes[lang], delim_pat=DELIM_PAT_NO_DANDA
        )
    return input_sentences


def initialize_model_and_tokenizer(ckpt_dir, quantization, attn_implementation):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    if attn_implementation == "flash_attention_2":
        if is_flash_attn_2_available() and is_flash_attn_greater_or_equal_2_10():
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        model.half()

    model.eval()

    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i: i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(
            batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations


def translate_paragraph(input_text, src_lang, tgt_lang, model, tokenizer, ip):
    input_sentences = split_sentences(input_text, src_lang)
    translated_text = batch_translate(
        input_sentences, src_lang, tgt_lang, model, tokenizer, ip)
    return " ".join(translated_text)


# Load model directly


attn_implementation = "eager"
quantization = ""
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def transform_text(ip: [str]) -> [str]:
    print("classifying", ip)
    tokenizer = AutoTokenizer.from_pretrained("Narasimha/hinglish-distilbert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "Narasimha/hinglish-distilbert")

    classifier = LanguageIdentifier.from_modelstring(
        langmodel, norm_probs=True)

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    v = pipe(ip[0])

    # print(v)

    if (v[0]['label'] == 'LABEL_1' or v[0]['label'] == 'LABEL_0'):
        prob = classifier.classify(ip[0])
        # print(prob)
        if (prob[0] in ['as', 'hi', 'bn', 'kK', 'gu', 'kn', 'ur', 'ml', 'mr', 'ne', 'or', 'pa', 'ta', 'te']):
            return ip

        en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(
            "ai4bharat/indictrans2-en-indic-dist-200M", quantization, attn_implementation)

        processor = IndicProcessor(inference=True)

        src_lang, tgt_lang = "eng_Latn", "hin_Deva"
        hi_translations = batch_translate(
            ip, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, processor)

        # for input_sentence, translation in zip(ip, hi_translations):
        #     print(f"{'en'}: {input_sentence}")
        #     print(f"{'hin'}: {translation}")
        return hi_translations
    else:
        return ip


def label_text(ip: str):
    tokenizer = AutoTokenizer.from_pretrained(
        "Hate-speech-CNERG/indic-abusive-allInOne-MuRIL")
    model = AutoModelForSequenceClassification.from_pretrained(
        "Hate-speech-CNERG/indic-abusive-allInOne-MuRIL")
    # model.to_bettertransformer()
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return pipe(ip)


start_time = time.time()

tf = transform_text(["behen ke lode gaandu"])
if label_text(tf[0])[0]['label'] == 'LABEL_1':
    print("ABUSIVE TEXT")
else:
    print("NOT ABUSIVE")
print("--- %s seconds ---" % (time.time() - start_time))
