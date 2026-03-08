q_file=/home/zangzeyuan/data/AMBER/query_discriminative_prepared.jsonl
answer_file_name=amber_discriminative

output_path=/work/zangzeyuan/llava-v1.5-7b/amber_evaluation_discriminative
model_path=liuhaotian/llava-v1.5-7b

if [ ! -d "$output_path" ]; then
  mkdir -p "$output_path"
  echo "mkdir path: $output_path"
else
  echo "path exist: $output_path"
fi

# python amber_data_prepare.py


CUDA_VISIBLE_DEVICES=7 python ../model_verifier.py \
     --model-path $model_path \
     --model-version llava \
     --model-vision openai/clip-vit-large-patch14-336 \
     --bf16 \
     --use_verifier \
     --image-folder /home/zangzeyuan/data/AMBER/image \
     --question-file $q_file \
     --answers-file $output_path/$answer_file_name.json \
     --temperature 0.2 \
     --max_length 1024 \

python inference.py \
    --inference_data $output_path/$answer_file_name.json \
    --word_association /home/zangzeyuan/data/AMBER/AMBER/data/relation.json \
    --safe_words /home/zangzeyuan/data/AMBER/AMBER/data/safe_words.txt \
    --annotation /home/zangzeyuan/data/AMBER/AMBER/data/annotations.json \
    --metrics /home/zangzeyuan/data/AMBER/AMBER/data/metrics.txt \
    --evaluation_type d