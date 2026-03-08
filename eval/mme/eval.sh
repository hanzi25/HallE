# output_path=/raid_sdd/whz/experiments/halle/evaluation/llava_v1_5
# CUDA_VISIBLE_DEVICES=2 python eval.py \
#     --model-path /raid_sdd/zzy/model/llava_1_5 \
#     --model-version llava \
#     --bf16 \
#     --mme-path /raid_sdd/whz/data/MME \
#     --output-path $output_path \

# cd /raid_sdd/whz/data/MME/eval_tool
# python calculation.py --results_dir $output_path/mme_evaluation/



output_path=/work/zangzeyuan/exp17_llava_13b_verifier_logits_scalar_frozen_1.0_sharegpt_9k_1ep_16bz_3e5/eval/mme
CUDA_VISIBLE_DEVICES=6 python eval.py \
    --model-path /work/zangzeyuan/exp17_llava_13b_verifier_logits_scalar_frozen_1.0_sharegpt_9k_1ep_16bz_3e5 \
    --model-version llava_verifier \
    --model-vision openai/clip-vit-large-patch14-336 \
    --bf16 \
    --use_verifier \
    --mme-path /home/zangzeyuan/data/MME \
    --output-path $output_path




# # cd /raid_sdd/zzy/data/MME/eval_tool
# python calculation.py \
#     --results_dir $output_path/mme_evaluation/


# output_path=/raid_sdd/zzy/model/InternVL-Chat-ViT-6B-Vicuna-7B/eval/mme2

# CUDA_VISIBLE_DEVICES=3 python eval.py \
#     --model-path /raid_sdd/zzy/model/InternVL-Chat-ViT-6B-Vicuna-7B \
#     --model-version llava \
#     --model-vision /raid_sdd/zzy/model/InternViT-6B-224px \
#     --bf16 \
#     --mme-path /raid_sdd/zzy/data/MME \
#     --output-path $output_path


# cd /raid_sdd/zzy/data/MME/eval_tool
python calculation.py \
    --results_dir $output_path/mme_evaluation/
