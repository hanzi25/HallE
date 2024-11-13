CUDA_VISIBLE_DEVICES=1 python chair.py \
    --cap_file /raid_sdd/whz/experiments/halle/train/exp4_llava_controller_23k_1ep_16bz_2e5/eval_plus1/llava_controller_1.0_1113205340.jsonl \
    --image_id_key image_id \
    --caption_key caption \
    --cache chair.pkl \
    --coco_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations \
    --save_path /raid_sdd/whz/experiments/halle/train/exp4_llava_controller_23k_1ep_16bz_2e5/eval_plus1/eval_CHAIR.json