CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python3 eval/model_controller.py \
            --model-path /raid_sdd/zzy/model/InternVL-Chat-ViT-6B-Vicuna-7B \
            --model-version llava \
            --model-vision /raid_sdd/zzy/model/InternViT-6B-224px \
            --bf16 \
            --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
            --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
            --output_folder /raid_sdd/zzy/model/InternVL-Chat-ViT-6B-Vicuna-7B/eval