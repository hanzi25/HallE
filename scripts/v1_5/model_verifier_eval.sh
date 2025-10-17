CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python3 eval/model_controller.py \
            --model-path /work/zangzeyuan/exp15_internvl_chat_vit_6b_vicuna_7b_verifier_logits_scalar_frozen_1.0_joint_6+3+1_1ep_16bz_3e5 \
            --model-version llava_verifier \
            --model-vision OpenGVLab/InternViT-6B-224px \
            --bf16 \
            --use_verifier \
            --alpha 1.0 \
            --gt_file_path /archive/private/zangzeyuan/data/coco/annotations/instances_val2014.json \
            --image_path /archive/private/zangzeyuan/data/coco/val2014 \
            --output_folder /work/zangzeyuan/exp15_internvl_chat_vit_6b_vicuna_7b_verifier_logits_scalar_frozen_1.0_joint_6+3+1_1ep_16bz_3e5/eval/1.0/

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python3 eval/model_controller.py \
#             --model-path /work/zangzeyuan/exp16_llava_7b_lora_joint_6+3+1_1ep_16bz_3e5_merged \
#             --model-version llava \
#             --bf16 \
#             --gt_file_path /archive/private/zangzeyuan/data/coco/annotations/instances_val2014.json \
#             --image_path /archive/private/zangzeyuan/data/coco/val2014 \
#             --output_folder /work/zangzeyuan/exp16_llava_7b_lora_joint_6+3+1_1ep_16bz_3e5_merged/eval6