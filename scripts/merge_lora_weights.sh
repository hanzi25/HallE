MODEL_PATH=/work/zangzeyuan/exp18_llava_7b_lora_rank64_joint_6+3+1_1ep_16bz_3e5
MODEL_BASE=liuhaotian/llava-v1.5-7b

CUDA_VISIBLE_DEVICES=1 python scripts/merge_lora_weights.py \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --save-model-path ${MODEL_PATH}_merged \