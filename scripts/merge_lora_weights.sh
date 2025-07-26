MODEL_PATH=/raid_sdd/zzy/experiments/halle/train/exp14_llava_lora_joint_6+3+1+1.5k_layer32_1ep_16bz_3e5
MODEL_BASE=/raid_sdd/zzy/model/llava_1_5

CUDA_VISIBLE_DEVICES=3 python eval/merge_lora_weights.py \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --save-model-path ${MODEL_PATH}_merged \