python chair.py \
    --cap_file /work/zangzeyuan/exp18_llava_7b_lora_rank8_joint_6+3+1_1ep_16bz_3e5_merged/eval/llava.jsonl \
    --image_id_key image_id \
    --caption_key caption \
    --cache chair.pkl \
    --coco_path /archive/private/zangzeyuan/data/coco/annotations \
    --save_path /work/zangzeyuan/exp18_llava_7b_lora_rank8_joint_6+3+1_1ep_16bz_3e5_merged/eval/eval_CHAIR.json