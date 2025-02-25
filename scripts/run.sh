pip3 install -e ".[train,cu12]"
pip3 install ninja
pip3 install flash-attn==2.7.0.post2 --no-build-isolation
python3 utils/dl_nltk.py
