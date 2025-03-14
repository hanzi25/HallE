CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python3 eval/model_controller.py \
            --model-path /raid_sdd/zzy/experiments/halle/train/exp7_llava_lora_sharegpt_50k_1ep_16bz_2e4_merged \
            --model-version llava \
            --bf16 \
            --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
            --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
            --output_folder /raid_sdd/zzy/experiments/halle/train/exp7_llava_lora_sharegpt_50k_1ep_16bz_2e4_merged/eval