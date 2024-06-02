#/bin/bash

python3 -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
python3 -m pip install nltk sacremoses pandas regex mock transformers>=4.33.2 mosestokenizer
python3 -c "import nltk; nltk.download('punkt')"
python3 -m pip install bitsandbytes scipy accelerate datasets
python3 -m pip install sentencepiece
git clone https://github.com/VarunGumma/IndicTransTokenizer
cd IndicTransTokenizer
python3 -m pip install --editable ./
cd $root_dir

root_dir=$(pwd)
echo "Setting up the environment in the $root_dir"

echo "Setup completed!"