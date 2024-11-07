CUDA_VISIBLE_DEVICES=1 python chair.py \
    --cap_file /raid_sdd/whz/experiments/halle/inference/llava/llava_117154433.jsonl \
    --image_id_key image_id \
    --caption_key caption \
    --cache chair.pkl \
    --coco_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations \
    --save_path /raid_sdd/whz/experiments/halle/inference/llava/eval_CHAIR.json