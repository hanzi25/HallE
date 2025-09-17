annotation_dir="/raid_sdd/zzy/data/pope"
pope_types=("coco_pope_adversarial" "coco_pope_popular" "coco_pope_random")

# output_path=/raid_sdd/whz/experiments/halle/evaluation/llava_v1_5/pope_evaluation
output_path=/raid_sdd/zzy/model/InternVL-Chat-ViT-6B-Vicuna-7B/eval/pope_evaluation2

if [ ! -d "$output_path" ]; then
  mkdir -p "$output_path"
  echo "mkdir path: $output_path"
else
  echo "path exist: $output_path"
fi

for type in "${pope_types[@]}"
do
    # CUDA_VISIBLE_DEVICES=0 python eval.py \
    #     --model-path /raid_sdd/zzy/model/llava_1_5 \
    #     --model-version llava \
    #     --bf16 \
    #     --image-folder /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
    #     --question-file $annotation_dir/$type.json \
    #     --answers-file $output_path/result_$type.json \
    #     --temperature 0 \

    CUDA_VISIBLE_DEVICES=3 python eval.py \
        --model-path /raid_sdd/zzy/model/InternVL-Chat-ViT-6B-Vicuna-7B \
        --model-version llava \
        --model-vision /raid_sdd/zzy/model/InternViT-6B-224px \
        --bf16 \
        --use_verifier \
        --image-folder /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
        --question-file $annotation_dir/$type.json \
        --answers-file $output_path/result_$type.json \
        --temperature 0.2 \

done

python evaluate.py \
  --annotation-dir $annotation_dir \
  --output-path $output_path \
