# output_path=/raid_sdd/whz/experiments/halle/evaluation/llava_v1_5
# CUDA_VISIBLE_DEVICES=2 python eval.py \
#     --model-path /raid_sdd/zzy/model/llava_1_5 \
#     --model-version llava \
#     --bf16 \
#     --mme-path /raid_sdd/whz/data/MME \
#     --output-path $output_path \

# cd /raid_sdd/whz/data/MME/eval_tool
# python calculation.py --results_dir $output_path/mme_evaluation/

output_path=/raid_sdd/whz/experiments/halle/train/exp5_llava_verifier_scalar_init_0.1_minus1_9k_1ep_4bz_3e5/eval
CUDA_VISIBLE_DEVICES=2 python eval.py \
    --model-path /raid_sdd/whz/experiments/halle/train/exp5_llava_verifier_scalar_init_0.1_minus1_9k_1ep_4bz_3e5 \
    --model-version llava_verifier \
    --bf16 \
    --use_verifier \
    --mme-path /raid_sdd/whz/data/MME \
    --output-path $output_path \

cd /raid_sdd/whz/data/MME/eval_tool
python calculation.py --results_dir $output_path/mme_evaluation/