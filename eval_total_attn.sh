model_path=/raid_sdd/whz/model/llava_1_5
CUDA_VISIBLE_DEVICES=0 python3 eval/model_controller.py \
            --model-path $model_path \
            --model-version llava \
            --bf16 \
            --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
            --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
            --output_folder $model_path/eval_attention

# model_path=/raid_sdd/whz/experiments/halle/train/exp5_llava_verifier_scalar_init_0.1_minus1_9k_1ep_4bz_3e5
# CUDA_VISIBLE_DEVICES=0 python3 eval/model_controller.py \
#             --model-path $model_path \
#             --model-version llava_verifier \
#             --bf16 \
#             --use_verifier \
#             --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
#             --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
#             --output_folder $model_path/eval_attention

# CUDA_VISIBLE_DEVICES=0 python chair.py \
#     --cap_file $model_path/eval/llava_verifier.jsonl \
#     --image_id_key image_id \
#     --caption_key caption \
#     --cache chair.pkl \
#     --coco_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations \
#     --save_path $model_path/eval/eval_CHAIR.json
