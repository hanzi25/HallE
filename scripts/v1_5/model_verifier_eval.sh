CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 eval/model_controller.py \
            --model-path /raid_sdd/zzy/experiments/halle/train/exp7_llava_verifier_scalar_frozen_init_1_sharegpt_9k_1ep_16bz_3e5  \
            --model-version llava_verifier \
            --bf16 \
            --use_verifier \
            --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
            --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
            --output_folder /raid_sdd/zzy/experiments/halle/train/exp7_llava_verifier_scalar_frozen_init_1_sharegpt_9k_1ep_16bz_3e5/eval/