CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python3 eval/model_controller.py \
            --model-path /raid_sdd/zzy/experiments/halle/train/exp13_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1_layer12_1ep_16bz_3e5 \
            --model-version llava_verifier \
            --bf16 \
            --use_verifier \
            --alpha 1.0 \
            --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
            --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
            --output_folder /raid_sdd/zzy/experiments/halle/train/exp13_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1_layer12_1ep_16bz_3e5/eval/1.0/

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python3 eval/model_controller.py \
#             --model-path /raid_sdd/zzy/model/llava_1_5 \
#             --model-version llava \
#             --bf16 \
#             --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
#             --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
#             --output_folder /raid_sdd/zzy/experiments/halle/train/exp10_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1k_1ep_16bz_3e5/eval/1.3/