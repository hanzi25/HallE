CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python generate_single.py \
    --model-path /raid_sdd/zzy/experiments/halle/train/exp5_llava_verifier_scalar_init_0.1_minus1_9k_1ep_4bz_3e5 \
    --model-version llava_verifier \
    --bf16 \
    --use_verifier \
    --image-file /raid_sdd/zzy/HallE/COCO_val2014_000000391895.jpg