pip3 install -e ".[train,cu12]"
pip3 install ninja
pip3 install flash-attn --no-build-isolation
python3 dl_nltk.py
