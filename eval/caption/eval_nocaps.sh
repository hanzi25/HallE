export JRE_HOME=/raid_sdb/home/zzy/miniforge3/envs/halle/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.171-8.b10.el6_9.x86_64/jre
export JAVA_HOME=$JRE_HOME
export PATH=$JRE_HOME/bin:$PATH
export CLASSPATH=.:$JRE_HOME/lib

java -version

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python eval_caption.py \
    --model-path /raid_sdd/zzy/experiments/halle/train/exp10_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1k_1ep_16bz_3e5 \
	--model-version llava_verifier \
	--bf16 \
	--use_verifier \
	--gt_file_path /raid_sdd/zzy/data/halle/NoCaps/toy_dataset.json \
	--image_path /raid_sdd/zzy/data/halle/NoCaps \
	--output_folder /raid_sdd/zzy/HallE/tmp/nocaps 


# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python eval_caption.py \
#   --model-path /raid_sdd/zzy/model/llava_1_5 \
# 	--model-version llava \
# 	--bf16 \
# 	--use_verifier \
# 	--gt_file_path /raid_sdd/zzy/data/halle/NoCaps/nocaps_val.json \
# 	--image_path /raid_sdd/zzy/data/halle/NoCaps \
# 	--output_folder raid_sdd/zzy/HallE/tmp/llava_eval/nocaps 