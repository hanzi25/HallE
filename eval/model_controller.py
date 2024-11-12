import argparse
import torch
from tqdm import tqdm
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import json
import random
import datetime

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def save_result(path, args, results):
    now = datetime.datetime.now()
    time = now.strftime("%H%M%S")
    current_time = f"{now.month}{now.day}{time}"

    if args.model_version == 'llava_controller':
        save_file = f"{path}/{args.model_version}_{args.sigma}_{current_time}.jsonl"
    else:
        save_file = f"{path}/{args.model_version}_{current_time}.jsonl"

    with open(save_file, "w") as file:
        for res in results:
            json.dump(res, file)
            file.write('\n')

def eval_model(args):

    # ========================================
    #             Model Initialization
    # ========================================
    model_path = args.model_path
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, args.model_version, args.model_vision)
    
    if args.model_version == 'llava_controller':
        model.sigma = args.sigma
    model = model.cuda()
    qs = args.query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    print("Query: ",qs)
    
    # conversation version
    conv_mode = args.conv_mode
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # output path
    path = args.output_folder
    # Create the folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # ========================================
    #            Load Evaluation File
    # ========================================

    def load_vg_evaluation_file(args, number=100):
        # annotation_file: args.gt_file_path 
        # image_path: args.image_path

        vg_path = args.gt_file_path
        vg_objects = json.load(open('%s/objects.json' %(vg_path)))
        vg_objects = vg_objects[:number]
        image_ids = [obj['image_id'] for obj in vg_objects]
        image_files = list()
        for id in tqdm(image_ids):
            image_file = f'{args.image_path}/images2/VG_100K_2/{id}.jpg'
            if not os.path.isfile(image_file):
                image_file = f'{args.image_path}/images/VG_100K/{id}.jpg'
            image_files.append(image_file)
        return image_files, image_ids


    def load_coco_evaluation_file(args, number=500):
        # annotation_file: args.gt_file_path ( /raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json )
        # image_path: args.image_path ()
        
        # load image
        img_files = os.listdir(args.image_path)

        # load annotation and build img_dict
        coco_instance_path = args.gt_file_path
        with open(coco_instance_path, 'r') as f:
            lines = f.readlines()
        coco_anns = json.loads(lines[0])
        img_dict = {}
        categories = coco_anns["categories"]
        category_dict = {int(c["id"]): c["name"] for c in categories}
        for img_info in coco_anns["images"]:
            img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}
            
        # for ann_info in coco_anns["annotations"]:
        #     img_dict[ann_info["image_id"]]["anns"].append(
        #         category_dict[ann_info["category_id"]]
        #     )
        
        # select image and build image_files        
        image_ids = list(img_dict.keys())[:number]
        image_files = list()
        for image_id in tqdm(image_ids):
            image_name = f'COCO_val2014_{str(image_id).zfill(12)}.jpg'
            if image_name in img_files:
                image_file = f'{args.image_path}/{image_name}'
                image_files.append(image_file)

        print("Total number of image is ", len(image_files))
        
        return image_files, image_ids

    # ========================================
    #      load image files and annotation
    # ========================================
    print("Start loading image files...")
    if 'coco' in args.image_path:
        image_files, image_ids = load_coco_evaluation_file(args)
    elif 'vg' in args.image_path:
        image_files, image_ids = load_vg_evaluation_file(args)
    else:
        print("Not support such image path: ", args.image_path)
        return

    # ========================================
    #             Inference
    # ========================================
    results = []
    for i in tqdm(range(len(image_files))):
        image_file = image_files[i]
        image_id = image_ids[i]

        image = load_image(image_file)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_length=1024,
                # max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        results.append({
            'image_id':image_id,
            'image_file':image_file, 
            'caption':outputs
            })
        print(image_id, outputs)

    save_result(path, args, results)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="llava") # llava & llava_controller & llava_verifier
    parser.add_argument("--model-vision", type=str, default="/raid_sdd/whz/model/clip_vit_large_patch14_336")
    parser.add_argument("--sigma", type=float, default=0)
    parser.add_argument("--gt_file_path", type=str, default='/raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json')
    parser.add_argument("--image_path", type=str, default='/raid_sdd/zzy/data/halle/coco/coco2014/val2014')
    parser.add_argument("--query", type=str, default="Describe this image as detailed as possible.")
    parser.add_argument("--conv-mode", type=str, default='v1')
    parser.add_argument("--output_folder", type=str, default='./')
    args = parser.parse_args()
    eval_model(args)
