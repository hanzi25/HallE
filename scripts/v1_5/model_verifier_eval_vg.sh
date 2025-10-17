CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python3 eval/model_controller.py \
            --model-path /work/zangzeyuan/exp10_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1k_1ep_16bz_3e5 \
            --model-version llava_verifier \
            --model-vision openai/clip-vit-large-patch14-336 \
            --bf16 \
            --use_verifier \
            --alpha 1.0 \
            --gt_file_path /archive/private/zangzeyuan/data/VG \
            --image_path /archive/private/zangzeyuan/data/VG \
            --output_folder /work/zangzeyuan/exp10_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1k_1ep_16bz_3e5/vg

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 eval/model_controller.py \
#             --model-path liuhaotian/llava-v1.5-7b \
#             --model-version llava \
#             --bf16 \
#             --gt_file_path /archive/private/zangzeyuan/data/VG \
#             --image_path /archive/private/zangzeyuan/data/VG \
#             --output_folder /work/zangzeyuan/llava-v1.5-7b/vg