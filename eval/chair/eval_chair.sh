python chair.py \
    --cap_file /work/zangzeyuan/exp16_llava_7b_lora_joint_6+3+1_1ep_16bz_3e5_merged/eval6/llava.jsonl \
    --image_id_key image_id \
    --caption_key caption \
    --cache chair.pkl \
    --coco_path /archive/private/zangzeyuan/data/coco/annotations \
    --save_path /work/zangzeyuan/exp16_llava_7b_lora_joint_6+3+1_1ep_16bz_3e5_merged/eval6/eval_CHAIR.json