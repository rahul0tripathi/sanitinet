from huggingface_hub import snapshot_download

cache_dir = "./snapshot"

MODEL_HINGLISH_CLASSIFIER = "Narasimha/hinglish-distilbert"
MODEL_INIDIC_TRANS = "ai4bharat/indictrans2-en-indic-dist-200M"
MODEL_HATE_SPEECH_CLASSIFIER = "Hate-speech-CNERG/indic-abusive-allInOne-MuRIL"


models = [
    MODEL_HINGLISH_CLASSIFIER,
    MODEL_INIDIC_TRANS,
    MODEL_HATE_SPEECH_CLASSIFIER
]


def get_hf_model(repo_name):
    return snapshot_download(repo_name, cache_dir=cache_dir)


def preload():
    [get_hf_model(repo_name) for repo_name in models]
