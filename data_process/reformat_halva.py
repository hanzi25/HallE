import json
import os

data_path = "/raid_sdd/whz/data/HALVA/data/data.json"
output_path = "/raid_sdd/zzy/data/halle/halva_instruct_21k.json"

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

converted_data = []

for item in data:
    new_item = {
        "id": str(item["id"]),
        "image": item["image"],
        "type": item["type"],
        "tag": item["tag"],
        "conversations": [
            {
                "from": "human",
                "value": item["question"]
            },
            {
                "from": "gpt",
                "value": item["correct_answer"]
            }
        ],
    }
    converted_data.append(new_item)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)

print(f"Conversion completed, output: {output_path}")