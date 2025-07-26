python chair.py \
    --cap_file /raid_sdd/zzy/experiments/halle/train/exp14_llava_lora_joint_6+3+1+1.5k_layer32_1ep_16bz_3e5_merged/eval/llava.jsonl \
    --image_id_key image_id \
    --caption_key caption \
    --cache chair.pkl \
    --coco_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations \
    --save_path /raid_sdd/zzy/experiments/halle/train/exp14_llava_lora_joint_6+3+1+1.5k_layer32_1ep_16bz_3e5_merged/eval/eval_CHAIR.json