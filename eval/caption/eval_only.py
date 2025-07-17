import sys
from pycocoevalcap_local.eval import COCOEvalCap
from pycocotools.coco import COCO
import os

# 设置环境变量
jre_home = "/raid_sdb/home/zzy/miniforge3/envs/halle/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.171-8.b10.el6_9.x86_64/jre"
os.environ["JRE_HOME"] = jre_home
os.environ["JAVA_HOME"] = jre_home
os.environ["PATH"] = f"{jre_home}/bin:" + os.environ.get("PATH", "")
os.environ["CLASSPATH"] = f".:{jre_home}/lib"


gt_file_path = '/raid_sdd/zzy/data/halle/NoCaps/nocaps_val.json'
result_file = '/raid_sdd/zzy/experiments/halle/train/exp12_llava_verifier_logits_scalar_frozen_1.0_joint_8+0.5+0.5_1ep_16bz_3e5/eval/nocaps/short/nocaps_val_llava_verifier.jsonl'

# COCO评估
coco = COCO(gt_file_path)
coco_result = coco.loadRes(result_file)
coco_eval = COCOEvalCap(coco, coco_result)
coco_eval.params["image_id"] = coco_result.getImgIds()
coco_eval.evaluate()

# 打印评估指标
for metric, score in coco_eval.eval.items():
    print(f"{metric}: {score:.3f}")