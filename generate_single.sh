CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python generate_single.py \
    --model-path /raid_sdi/home/zzy/experiments/halle/train/exp10_llava_verifier_logits_scalar_frozen_1.0_sharegpt_9k_1ep_16bz_3e5 \
    --model-version llava_verifier \
    --bf16 \
    --use_verifier \
    --image-file /raid_sdi/home/zzy/HallE/COCO_val2014_000000391895.jpg