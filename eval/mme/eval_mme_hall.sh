# output_path=/raid_sdd/zzy/model/llava_1_5/eval/mme_hall2

# CUDA_VISIBLE_DEVICES=2 python eval.py \
#     --model-path /raid_sdd/zzy/model/llava_1_5 \
#     --model-version llava \
#     --bf16 \
#     --mme-path /raid_sdd/zzy/data/MME \
#     --output-path $output_path


# python calculation.py \
#     --results_dir $output_path/mme_evaluation/ \
#     --save_name "mme_hall_evaluation_results.json" \
#     --mme_hall

output_path=/raid_sdd/zzy/experiments/halle/train/exp15_internvl_chat_vit_6b_vicuna_7b_verifier_logits_scalar_frozen_1.0_joint_6+3+1_1ep_16bz_3e5/eval/mme2

# CUDA_VISIBLE_DEVICES=2 python eval.py \
#     --model-path /raid_sdd/zzy/experiments/halle/train/exp10_llava_verifier_logits_scalar_frozen_1.0_sharegpt_9k_1ep_16bz_3e5 \
#     --model-version llava_verifier \
#     --model-vision /raid_sdd/zzy/model/InternViT-6B-224px \
#     --bf16 \
#     --use_verifier \
#     --mme-path /raid_sdd/zzy/data/MME \
#     --output-path $output_path


python calculation.py \
    --results_dir $output_path/mme_evaluation/ \
    --save_name "mme_hall_evaluation_results.json" \
    --mme_hall