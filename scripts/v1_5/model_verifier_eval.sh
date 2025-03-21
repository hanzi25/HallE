CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python3 eval/model_controller.py \
            --model-path /raid_sdi/home/zzy/experiments/halle/train/exp10_llava_verifier_logits_scalar_frozen_1.0_sharegpt_9k_1ep_16bz_3e5 \
            --model-version llava_verifier \
            --bf16 \
            --use_verifier \
            --gt_file_path /raid_sdi/home/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
            --image_path /raid_sdi/home/zzy/data/halle/coco/coco2014/val2014 \
            --output_folder /raid_sdi/home/zzy/experiments/halle/train/exp10_llava_verifier_logits_scalar_frozen_1.0_sharegpt_9k_1ep_16bz_3e5/eval/