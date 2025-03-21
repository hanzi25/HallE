python utils/chair.py \
    --cap_file /raid_sdi/home/zzy/experiments/halle/train/exp7_llava_verifier_scalar_frozen_1.0_sharegpt_5k_1ep_16bz_3e5/eval/1.0/llava_verifier.jsonl \
    --image_id_key image_id \
    --caption_key caption \
    --cache chair.pkl \
    --coco_path /raid_sdi/home/zzy/data/halle/coco/coco2014/annotations \
    --save_path /raid_sdi/home/zzy/experiments/halle/train/exp7_llava_verifier_scalar_frozen_1.0_sharegpt_5k_1ep_16bz_3e5/eval/1.0/eval_CHAIR.json