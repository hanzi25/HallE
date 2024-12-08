# CUDA_VISIBLE_DEVICES=1 python3 eval/model_controller.py \
#             --model-path /raid_sdd/whz/model/llava_1_5 \
#             --model-version llava \
#             --gt_file_path /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
#             --image_path /raid_sdd/zzy/data/halle/coco/coco2014/val2014 \
#             --output_folder /raid_sdd/whz/experiments/halle/inference/llava

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 eval/model_controller.py \
            --model-path /raid_sdi/home/zzy/experiments/halle/train/exp5_llava_verifier_scalar_init_1.0_minus1_9k_1ep_4bz_3e5/ \
            --model-version llava_verifier \
            --use_verifier False \
            --bf16 True \
            --gt_file_path /raid_sdi/home/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json \
            --image_path /raid_sdi/home/zzy/data/halle/coco/coco2014/val2014 \
            --output_folder /raid_sdi/home/zzy/experiments/halle/train/exp5_llava_verifier_scalar_init_1.0_minus1_9k_1ep_4bz_3e5/eval
