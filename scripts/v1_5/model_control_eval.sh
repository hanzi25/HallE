# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python3 eval/model_controller.py \
#             --model-path /raid_sdi/home/zzy/model/llava_1_5 \
#             --model-version llava \
#             --gt_file_path /raid_sdi/home/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
#             --image_path /raid_sdi/home/zzy/data/halle/coco/coco2014/val2014 \
#             --output_folder /raid_sdi/home/zzy/experiments/halle/inference/llava

# CUDA_VISIBLE_DEVICES=2 python3 eval/model_controller.py \
#             --model-path /raid_sdd/whz/experiments/halle/train/exp2_llava_verifier_minus1_9k_1ep_16bz_2e5/ \
#             --model-version llava_verifier \
#             --bf16 True \
#             --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
#             --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
#             --output_folder /raid_sdd/whz/experiments/halle/train/exp2_llava_verifier_minus1_9k_1ep_16bz_2e5/eval

# CUDA_VISIBLE_DEVICES=2 python3 eval/model_controller.py \
#             --model-path /raid_sdd/whz/experiments/halle/train/exp4_llava_controller_23k_1ep_16bz_2e5/ \
#             --model-version llava_controller \
#             --bf16 True \
#             --sigma 1 \
#             --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
#             --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
#             --output_folder /raid_sdd/whz/experiments/halle/train/exp4_llava_controller_23k_1ep_16bz_2e5/eval_plus1