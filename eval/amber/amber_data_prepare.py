import json
import os
from PIL import Image
from tqdm import tqdm

def check_image(image_file, dd):
    try:
        with Image.open(image_file).convert('RGB') as img:
            pass
    except Exception as e:
        return (dd['image'], str(e))

if __name__ == "__main__":
    file = "/home/whz/code/AMBER/data/query/query_generative.json"
    image_path = "/raid_sdd/whz/data/AMBER/image"

    with open(file, "r") as f:
        questions = eval(f.read())

    error = list()
    data = list()
    for dd in tqdm(questions):
        tmp = {
            'question_id': dd['id'],
            'image': dd['image'],
            'text': dd['query'],
            'label': '',
        }
        image_file = os.path.join(image_path, tmp['image'])
        check = check_image(image_file, tmp)

        if check is None:
            data.append(tmp)
            continue
        else:
            print(check)
            error.append(dd)
    
    print("Error: ", len(error))

    with open("/raid_sdd/whz/data/AMBER/query_generative_prepared.jsonl", "w") as f:
        for dd in data:
            f.write(json.dumps(dd)+'\n')