# Hallucination Control

## Reference

[[HallE-Control: Controlling Object Hallucination in Large Mutimodal Models](https://arxiv.org/pdf/2310.01779v3.pdf)] [[Project Page](https://bohanzhai.github.io/halle-switch.github.io/)] <br>

Paper:
- [3/27] 🔥 We released the new version **HallE-Control: Controlling Object Hallucination in LMMs**. Checkout the [paper](https://arxiv.org/pdf/2310.01779v3.pdf).
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
```Shell

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

- Generate captions for images.
```Shell
bash scripts/v1_5/model_control_eval.sh
```


- Calculate CHAIR score.
```Shell
bash eval_chair.sh
```


# Others

1. 显存占用
纯推理显存占用：15802MiB 