# Hallucination Control

## Reference

[[HallE-Control: Controlling Object Hallucination in Large Mutimodal Models](https://arxiv.org/pdf/2310.01779v3.pdf)] [[Project Page](https://bohanzhai.github.io/halle-switch.github.io/)] <br>

Paper:
- [3/27] 🔥 We released the new version **HallE-Control: Controlling Object Hallucination in LMMs**. Checkout the [paper](https://arxiv.org/pdf/2310.01779v3.pdf).
- [12/3] 🔥 We released **HallE-Switch: Controlling Object Hallucination in LVLMs**. Checkout the [paper](https://arxiv.org/abs/2310.01779).

## Contents
- [Install](#install)
- [Training](#training)
- [Evaluation](#evaluation)

## Install
1. Clone this repository and navigate to HallE_Control folder
```bash
git clone https://github.com/bronyayang/HallE_Control.git
cd HallE_Control
```

2. Install Package
```Shell
conda create -n halle python=3.10 -y
conda activate halle
bash scripts/run.sh
```

3. Install nltk
```Shell
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
```


## Training

### Train Controller

1. Prepare data

- Follow [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning) to prepare data.
Download controller data [here](https://drive.google.com/drive/folders/1ZxRE2BNVgWXNSjPv5fv6gw4JzwKeXU4b?usp=sharing) and put in ./data folder.

- data_file: /raid_sdd/whz/data/halle/detail_switch_23k.json

2. Start training

- Train controller: Model can output less hallucinated/more imagination captions based on $\epsilon$

```Shell
bash scripts/v1_5/tune_controller.sh
```
Make sure the output_dir contains the word "llava" and "controller" for correct inference behavior.

### Train Vision Verifier

1. Prepare data

- Extract samples whose caption has grounded objects.("hall_factor"=-1)

- data_file: /raid_sdd/whz/data/halle/detail_switch_minus_1_9093.json

2. Start trainning

- Train vision verifier: llava/model/language_model/llava_llama_verifier.py

```Shell
bash scripts/v1_5/tune_verifier.sh
```


## Evaluation

1. CHAIR

- Generate captions for images.

```Shell
bash scripts/v1_5/model_control_eval.sh
```

- Calculate CHAIR score.
```Shell
bash eval_chair.sh
```


# Others

1. 训练时间 & 显存占用
- llava verifier 训练：
    - 两卡：50 min 
    - 33422MiB / 49140MiB 
    - 22810MiB / 46068MiB


2. 推理时间 & 现存占用
- llava 纯推理显存占用：15802MiB 
