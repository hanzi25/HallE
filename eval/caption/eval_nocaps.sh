CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python eval_caption.py \
    --model-path /raid_sdd/zzy/experiments/halle/train/exp10_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1k_1ep_16bz_3e5 \
	--model-version llava_verifier \
	--bf16 \
	--use_verifier \
	--gt_file_path /raid_sdd/zzy/data/halle/NoCaps/nocaps_val.json \
	--image_path /raid_sdd/zzy/data/halle/NoCaps \
	--output_folder /raid_sdd/zzy/experiments/halle/train/exp10_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1k_1ep_16bz_3e5/eval/nocaps 


# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python eval_caption.py \
#   --model-path /raid_sdd/zzy/model/llava_1_5 \
# 	--model-version llava \
# 	--bf16 \
# 	--use_verifier \
# 	--gt_file_path /raid_sdd/zzy/data/halle/NoCaps/nocaps_val.json \
# 	--image_path /raid_sdd/zzy/data/halle/NoCaps \
# 	--output_folder raid_sdd/zzy/HallE/tmp/llava_eval/nocaps 