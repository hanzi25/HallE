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
- [Experiments](#experiments)
- [Results](#results)
- [Others](#others)

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

2. Start training

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

# Experiments

1. LLaVA
    - Path: /raid_sdd/whz/experiments/halle/inference/llava
    - LLaVA 原始 ckpt 推理 
2. LLaVA controller:
    - Path: exp4_llava_controller_23k_1ep_16bz_2e5
    - 复现 halle-control
3. Verifier_v1:
    - Path: exp2_llava_verifier_minus1_9k_1ep_16bz_2e5
    - 训练时 hidden states（system_prompt + vision_embeds + question + answer） 所有部分都和 vision_embeds 进行 cross attention
4. Verifier_v2:
    - Path: exp3_llava_verifier_minus1_9k_1ep_8bz_2e5
    - 训练时 hidden states 中仅 question + answer 与 vision_embeds 进行 cross attention

...

实验记录：https://kdocs.cn/l/cmqOJvKsbG1j

# Results

| Model | Tune | Alpha Type | Device & bz| Train Data | lr | CHAIRs | CHAIRi | Recall | Len | max length | max new tokens |
| ---   | ---: |       ---: |       ---: |       ---: | ---: | ---:   | ---:   | ---:   | ---: | ---: | ---: |
| LLaVA-v1.5 |  |  |  |   |   | 0.47 | 0.1386 | 0.7814 | 1.00922 | 1024 |  |
| LLaVA | Controller | sigma=-1 | 2x16bz | 23k | 2e5 | 0.434 | 0.1236 | 0.7565 | 1.13142 | 1024 |  |
| LLaVA | Controller | sigma=+1| 2x16bz | 23k | 2e5 | 0.518 | 0.1599 | 0.7792 | 0.98124 | 1024 |  |
| LLaVA | Controller | sigma=-1| 2x8bz | 9k(-1) | 2e5 | 0.394 | 0.1156 | 0.71398 | 1.31516 | 1024 |  |
| LLaVA | Verifier_v1 | vector | 2x8bz |  9k(-1) | 2e5 | 0.454 | 0.1307 | 0.7572 | 1.06614 | 1024 |  |
| LLaVA | Verifier_v2 | vector | 2x8bz | 9k(-1) | 2e5 | 0.448 | 0.1255 | 0.7643 | 1.06818 |1024 | |
| LLaVA | Verifier_v2 | vector | 2x8bz | 9k(-1) | 2e5 | 0.428 | 0.1268 | 0.7622 | 1.0791 | | 200 |
| LLaVA | Verifier_v2 | vector | 2x8bz | 9k(-1) | 2e5 | 0.346| 0.109 | 0.7168 | 0.8437 | | 100 |
| LLaVA | Verifier_v2 | vector | 2x8bz | 9k(-1) | 1e5 | 0.458 | 0.1343 | 0.7743 | 1.03144 |1024 | |
| LLaVA | Verifier_v2 | vector | 2x8bz | 9k(-1) | 3e5 | 0.436 | 0.1278 | 0.7629 | 1.12096 |1024 | |
| LLaVA | Verifier_v2 | matrix | 4x4bz | 9k(-1) | 3e5 | 0.392 | 0.1203 | 0.7317 | 1.22746 |1024 | |
| LLaVA | Verifier_v2 | vector | 4x4bz | 9k(-1) | 3e5 | 0.462 | 0.1320 | 0.7615 | 1.1607 |1024 | |
| LLaVA | Verifier_v2 | scalar(0) | 2x8bz | 9k(-1) | 3e5 | 0.446 | 0.1250 | 0.7615 | 1.0407 |1024 | |
| LLaVA | Verifier_v2 | scalar(0.1) | 1x4bz | 9k(-1) | 3e5 | 0.392 | 0.1113 | 0.7402 | 1.17946 |1024 | |





# Inference Config
```Shell
    do_sample=True, # Nucleus
    temperature=0.2,
```

# Others

1. 训练时间 & 显存占用
- llava verifier 训练（两卡）：
    - 时间：50 min 
    - 33422MiB / 49140MiB 
    - 22810MiB / 46068MiB
- llava verifier 训练（一卡）：
    - 4bz大概是26G-28G，8bz有时候能到40多


2. 推理时间 & 显存占用
- llava 纯推理显存占用：15802MiB 
- 单卡推理 4.65s/it;
- CHAIR评测时间 40min

3. Deepspeed 
[Deepspeed zero3 debug](https://deepspeed.readthedocs.io/en/latest/zero3.html#debugging) 查看梯度
```Shell
deepspeed.utils.safe_get_full_grad(model.alpha.weight)
```

4. 参考链接
- [OPERA CHAIR](https://github.com/shikiw/OPERA/blob/main/chair_eval.py)
- [文件开权限，chmod 777](https://zhuanlan.zhihu.com/p/705959942)
- [hf-mirror huggingface 下载模型和数据](https://hf-mirror.com/)
- [ssh 权限](https://zhuanlan.zhihu.com/p/688103044)
- [Python调试器pdb教程](https://zhuanlan.zhihu.com/p/37294138)

