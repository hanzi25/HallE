q_file=/raid_sdd/zzy/data/AMBER/query_all_prepared.jsonl
answer_file_name=amber_all

output_path=/raid_sdd/zzy/experiments/halle/train/exp10_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1k_1ep_16bz_3e5/eval/amber_evaluation2
model_path=/raid_sdd/zzy/experiments/halle/train/exp10_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1k_1ep_16bz_3e5

if [ ! -d "$output_path" ]; then
  mkdir -p "$output_path"
  echo "mkdir path: $output_path"
else
  echo "path exist: $output_path"
fi

# python amber_data_prepare.py


# CUDA_VISIBLE_DEVICES=2 python ../model_verifier.py \
#      --model-path $model_path \
#      --model-version llava \
#      --model-vision /raid_sdd/zzy/model/InternViT-6B-224px \
#      --bf16 \
#      --use_verifier \
#      --image-folder /raid_sdd/zzy/data/AMBER/image \
#      --question-file $q_file \
#      --answers-file $output_path/$answer_file_name.json \
#      --temperature 0.2 \
#      --max_length 1024 \

python inference.py \
    --inference_data $output_path/$answer_file_name.json \
    --word_association /raid_sdd/zzy/data/AMBER/AMBER/data/relation.json \
    --safe_words /raid_sdd/zzy/data/AMBER/AMBER/data/safe_words.txt \
    --annotation /raid_sdd/zzy/data/AMBER/AMBER/data/annotations.json \
    --metrics /raid_sdd/zzy/data/AMBER/AMBER/data/metrics.txt \
    --evaluation_type a