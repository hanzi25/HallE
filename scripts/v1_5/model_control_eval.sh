
CUDA_VISIBLE_DEVICES=3 python3 eval/model_controller.py \
            --model-path /work/zangzeyuan/exp16_llava_7b_controller_joint_6+3+1_1ep_16bz_3e5 \
            --model-version llava_controller \
            --bf16 \
            --sigma -1 \
            --gt_file_path /archive/private/zangzeyuan/data/coco/annotations/instances_val2014.json \
            --image_path /archive/private/zangzeyuan/data/coco/val2014 \
            --output_folder /work/zangzeyuan/exp16_llava_7b_controller_joint_6+3+1_1ep_16bz_3e5/eval5