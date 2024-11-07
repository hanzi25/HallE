CUDA_VISIBLE_DEVICES=0 python3 eval/model_controller.py \
            --model-path LLAVA_CONTROLLER_MODEL_PATH \
            --model-version llava \
            --gt_file_path ./data/VisualGenome_task \
            --image_path ./data \
            --output_folder OUTPUT_PATH


CUDA_VISIBLE_DEVICES=0 python3 eval/model_controller.py \
            --model-path LLAVA_CONTROLLER_MODEL_PATH \
            --model-version llava_control \
            --sigma 0 \
            --gt_file_path ./data/VisualGenome_task \
            --image_path ./data \
            --output_folder OUTPUT_PATH

