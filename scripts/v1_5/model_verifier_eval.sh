# CUDA_VISIBLE_DEVICES=1 python3 eval/model_controller.py \
#             --model-path /raid_sdd/whz/model/llava_1_5 \
#             --model-version llava \
#             --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
#             --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
#             --output_folder /raid_sdd/whz/experiments/halle/inference/llava

CUDA_VISIBLE_DEVICES=0 python3 eval/model_controller.py \
            --model-path /raid_sdd/zzy/experiments/halle/train/exp5_llava_verifier_scalar_minus1_9k_1ep_8bz_3e5/ \
            --model-version llava_verifier \
            --use_verifier True \
            --bf16 True \
            --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
            --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
            --output_folder /raid_sdd/whz/experiments/halle/train/exp3_llava_verifier_minus1_9k_1ep_8bz_1e5/eval
