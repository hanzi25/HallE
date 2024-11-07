# HallE-Control: Controlling Object Hallucination in Large Mutimodal Models

[[Paper](https://arxiv.org/pdf/2310.01779v3.pdf)] [[Project Page](https://bohanzhai.github.io/halle-switch.github.io/)] <br>
[Bohan Zhai*](https://www.linkedin.com/in/bohan-zhai-202507154/), [Shijia Yang*](https://bronyayang.github.io/personal_website/), [Chenfeng Xu](https://www.chenfengx.com/), [Sheng Shen](https://sincerass.github.io/), [Kurt Keutzer](https://people.eecs.berkeley.edu/~keutzer/), [Chunyuan Li](https://chunyuan.li/), [Manling Li](https://limanling.github.io/)


## Release
- [3/28] We release the full training and eval code. Welcome any question!
- [3/27] 🔥 We released the new version **HallE-Control: Controlling Object Hallucination in LMMs**. Checkout the [paper](https://arxiv.org/pdf/2310.01779v3.pdf).
- [12/13] We add CCEval's code for evaluation object existence hallucination.
- [12/3] 🔥 We released **HallE-Switch: Controlling Object Hallucination in LVLMs**. Checkout the [paper](https://arxiv.org/abs/2310.01779).

## Contents
- [Install](#install)
- [Data](#data)
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

## Data

1. ln data
```Shell
conda create -n halle python=3.10 -y
conda activate halle
bash scripts/run.sh
```

## Training

1. Prepare data

Follow [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning) to prepare data.
Download controller data [here](https://drive.google.com/drive/folders/1ZxRE2BNVgWXNSjPv5fv6gw4JzwKeXU4b?usp=sharing) and put in ./data folder.

2. Start training

- Train controller: Model can output less hallucinated/more imagination captions based on $\epsilon$

```Shell
bash scripts/v1_5/tune_controller.sh
```
Make sure the output_dir contains the word "controller" for correct inference behavior.

- Train indication: Model can output caption with [object] indication on imagined objects

```Shell
bash scripts/v1_5/finetune_indication.sh
```

## Evaluation

1. CHAIR

```Shell
python3 cceval.py --cap_file [YOUR_CAPTION_FILE_PATH] --key [YOUR_OPENAI_API_KEY]
```
