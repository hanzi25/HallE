import json
from collections import defaultdict
import os

input_file = "/home/zangzeyuan/data/halle/flickr30k_karpathy_train.json"
output_file = "/home/zangzeyuan/data/halle/flickr30k_karpathy_train_conversation.json"

prompts = [
    "Describe the following image.\n<image>",
    "What do you see in this image?\n<image>",
    "Explain what is happening in the picture.\n<image>",
    "Can you describe this image in detail?\n<image>",
    "What is going on in this image?\n<image>",
    "Give a detailed description of the image.\n<image>",
    "Summarize the scene shown in the image.\n<image>"
]

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

groups = defaultdict(list)
for ann in data["annotations"]:
    groups[ann["image_id"]].append(ann)

results = []

for image_id, items in groups.items():
    image_path = items[0]["image"]
    image_file = os.path.basename(image_path)

    for i, ann in enumerate(items):
        caption = ann["caption"]
        prompt = prompts[i % len(prompts)]  # 轮换使用 prompt

        sample = {
            "id": image_id,          # 不补零
            "image": image_file,
            "conversations": [
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt",
                    "value": caption
                }
            ]
        }

        results.append(sample)

# 可选：按 (image_id, 原 annotation id) 排序，保证稳定顺序
results.sort(key=lambda x: (x["id"]))

with open(output_file, "w", encoding="utf-8") as fout:
    json.dump(results, fout, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} samples to {output_file}")