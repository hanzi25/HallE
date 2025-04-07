CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python generate_single.py \
    --model-path /raid_sdd/zzy/experiments/halle/train/exp10_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1k_1ep_16bz_3e5 \
    --model-version llava_verifier \
    --bf16 \
    --use_verifier \
    --image-file /raid_sdd/zzy/HallE/COCO_val2014_000000391895.jpg