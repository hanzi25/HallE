# python chair.py \
#     --cap_file /raid_sdi/home/zzy/experiments/halle/train/exp5_llava_verifier_scalar_init_1.0_minus1_9k_1ep_4bz_3e5/eval/llava_verifier.jsonl \
#     --image_id_key image_id \
#     --caption_key caption \
#     --cache chair.pkl \
#     --coco_path /raid_sdi/home/zzy/data/halle/coco/coco2014/annotations \
#     --save_path /raid_sdi/home/zzy/experiments/halle/train/exp5_llava_verifier_scalar_init_1.0_minus1_9k_1ep_4bz_3e5/eval/eval_CHAIR.json\

python chair.py \
    --cap_file example.jsonl \
    --image_id_key image_id \
    --caption_key caption \
    --cache chair.pkl \
    --coco_path /raid_sdi/home/zzy/data/halle/coco/coco2014/annotations \
    --save_path eval_CHAIR.json\