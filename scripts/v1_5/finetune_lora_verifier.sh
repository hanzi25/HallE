deepspeed --include localhost:0 --master_port 25432 llava/train/train_with_verifier.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /raid_sdd/zzy/model/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /raid_sdd/zzy/data/halle/llava_v1_5_mix665k.json \
    --image_folder /raid_sdd/zzy/data/halle \
    --vision_tower /raid_sdd/zzy/model/clip_vit_large_patch14_336 \
    --pretrain_mm_mlp_adapter /raid_sdd/zzy/model/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /raid_sdd/zzy/experiments/halle/instruct/exp1_llava_vicuna_7b_v1.5_lora_with_verifier_mix665k_1ep_8bz_2e4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --wandb_project_name "llava_with_verifier"