# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python3 eval/model_controller.py \
#             --model-path /work/zangzeyuan/exp21_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1_flickr_1k_single_1ep_16bz_3e5 \
#             --model-version llava_verifier \
#             --model-vision openai/clip-vit-large-patch14-336 \
#             --bf16 \
#             --use_verifier \
#             --alpha 0.2 \
#             --gt_file_path /archive/private/zangzeyuan/data/coco/annotations/instances_val2014.json \
#             --image_path /archive/private/zangzeyuan/data/coco/val2014 \
#             --output_folder /work/zangzeyuan/exp21_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1_flickr_1k_single_1ep_16bz_3e5/eval/0.2/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python3 eval/model_controller.py \
            --model-path /work/zangzeyuan/exp18_llava_7b_lora_rank64_joint_6+3+1_1ep_16bz_3e5_merged/ \
            --model-version llava \
            --bf16 \
            --gt_file_path /archive/private/zangzeyuan/data/coco/annotations/instances_val2014.json \
            --image_path /archive/private/zangzeyuan/data/coco/val2014 \
            --output_folder /work/zangzeyuan/exp18_llava_7b_lora_rank64_joint_6+3+1_1ep_16bz_3e5_merged/eval