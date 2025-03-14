python utils/chair.py \
    --cap_file /raid_sdd/zzy/experiments/halle/train/exp7_llava_lora_sharegpt_50k_1ep_16bz_2e4_merged/eval/llava.jsonl \
    --image_id_key image_id \
    --caption_key caption \
    --cache chair.pkl \
    --coco_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations \
    --save_path /raid_sdd/zzy/experiments/halle/train/exp7_llava_lora_sharegpt_50k_1ep_16bz_2e4_merged/eval/eval_CHAIR.json