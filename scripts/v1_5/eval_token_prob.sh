CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python utils/get_prob_score.py \
    --model-path /raid_sdd/zzy/experiments/halle/train/exp12_llava_verifier_logits_scalar_frozen_1.0_joint_8+0.5+0.5_1ep_16bz_3e5 \
    --model-version llava_verifier \
    --bf16 \
    --alpha 1.0 \
    --use_verifier \
    --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
    --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
    --annotation_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations \
    --output_folder /raid_sdd/zzy/experiments/halle/train/exp12_llava_verifier_logits_scalar_frozen_1.0_joint_8+0.5+0.5_1ep_16bz_3e5/eval/prob_score/1.0
    
    #--output_folder /raid_sdd/zzy/HallE/tmp/temp=1.2_p=0.999_100/