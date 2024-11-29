model_path=/raid_sdd/whz/experiments/halle/train/exp5_llava_verifier_vector_minus1_9k_1ep_4bz_3e5
CUDA_VISIBLE_DEVICES=0 python3 eval/model_controller.py \
            --model-path /raid_sdd/zzy/experiments/halle/train/exp5_llava_verifier_vector_minus1_9k_1ep_4bz_3e5 \
            --model-version llava_verifier \
            --bf16 True \
            --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
            --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
            --output_folder $model_path/eval

CUDA_VISIBLE_DEVICES=0 python chair.py \
    --cap_file $model_path/eval/llava_verifier.jsonl \
    --image_id_key image_id \
    --caption_key caption \
    --cache chair.pkl \
    --coco_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations \
    --save_path $model_path/eval/eval_CHAIR.json


# model_path=/raid_sdd/whz/experiments/halle/train/exp4_llava_controller_minus1_9k_1ep_8bz_2e5
# CUDA_VISIBLE_DEVICES=1 python3 eval/model_controller.py \
#             --model-path $model_path \
#             --model-version llava_controller \
#             --bf16 True \
#             --sigma -1 \
#             --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
#             --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
#             --output_folder $model_path/eval

# CUDA_VISIBLE_DEVICES=1 python chair.py \
#     --cap_file $model_path/eval/llava_controller.jsonl \
#     --image_id_key image_id \
#     --caption_key caption \
#     --cache chair.pkl \
#     --coco_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations \
#     --save_path $model_path/eval/eval_CHAIR.json