CUDA_VISIBLE_DEVICES=1 python3 eval/model_controller.py \
            --model-path /raid_sdd/whz/model/llava_1_5 \
            --model-version llava \
            --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
            --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
            --output_folder /raid_sdd/whz/experiments/halle/inference/llava


# CUDA_VISIBLE_DEVICES=0 python3 eval/model_controller.py \
#             --model-path LLAVA_CONTROLLER_MODEL_PATH \
#             --model-version llava_control \
#             --sigma 0 \
#             --gt_file_path ./data/VisualGenome_task \
#             --image_path ./data \
#             --output_folder OUTPUT_PATH

