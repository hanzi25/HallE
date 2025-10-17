MODEL_PATH=/work/zangzeyuan/exp16_llava_7b_lora_joint_6+3+1_1ep_16bz_3e5
MODEL_BASE=liuhaotian/llava-v1.5-7b

CUDA_VISIBLE_DEVICES=3 python scripts/merge_lora_weights.py \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --save-model-path ${MODEL_PATH}_merged \