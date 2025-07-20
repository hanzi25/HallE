python chair.py \
    --cap_file /raid_sdd/zzy/experiments/halle/train/exp13_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1_layer12_1ep_16bz_3e5/eval/1.0/llava_verifier.jsonl \
    --image_id_key image_id \
    --caption_key caption \
    --cache chair.pkl \
    --coco_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations \
    --save_path /raid_sdd/zzy/experiments/halle/train/exp13_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1_layer12_1ep_16bz_3e5/eval/1.0/eval_CHAIR.json