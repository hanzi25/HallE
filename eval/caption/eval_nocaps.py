import argparse
import torch
from tqdm import tqdm
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
import requests
from PIL import Image
from io import BytesIO
import json
import random

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def save_result(path, args, results):

    if args.model_version == 'llava_controller':
        save_file = f"{path}/nocaps_val_{args.model_version}_{args.sigma}.jsonl"
    elif args.model_version == 'llava_verifier':
        if not args.use_verifier:
            save_file = f"{path}/nocaps_val_{args.model_version}_no_verifier.jsonl"
        else:
            save_file = f"{path}/nocaps_val_{args.model_version}.jsonl"
    else:
        save_file = f"{path}/nocaps_val_{args.model_version}.jsonl"

    with open(save_file, "w") as file:
        json.dump(results, file)

    return save_file

def eval_model(args):

    # ========================================
    #             Model Initialization
    # ========================================
    model_path = args.model_path
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, args.model_version, args.model_vision, load_bf16=args.bf16)
    
    if args.model_version == 'llava_controller':
        model.sigma = args.sigma
    elif args.model_version == 'llava_verifier':
        if not args.use_verifier:
            model.alpha = torch.nn.Parameter(torch.tensor(0.0))
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

    def load_nocaps_evaluation_file(args):
        # annotation_file: args.gt_file_path 
        # image_path: args.image_path

        nocaps_instance_path = args.gt_file_path
        with open(nocaps_instance_path, 'r') as f:
            tmp_data = json.load(f)
        new_tmp_data = tmp_data['annotations']
        image_ids = list()
        image_files = list()
        for item in new_tmp_data:
            image_id = item['image_id']
            if image_id not in image_ids: image_ids.append(image_id)
            image_file = f'{args.image_path}/{item["image"]}'
            if image_file not in image_files: image_files.append(image_file)
        return list(image_files), list(image_ids)

    # ========================================
    #      load image files and annotation
    # ========================================
    print("Start loading image files...")
    image_files, image_ids = load_nocaps_evaluation_file(args)

    # ========================================
    #             Inference
    # ========================================
    
    results = []
    for i in tqdm(range(len(image_files))):
        image_file = image_files[i]
        image_id = image_ids[i]
        
        # import pdb; pdb.set_trace()

        image = load_image(image_file)
        if args.bf16:
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(torch.bfloat16).cuda()
        else:
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
                stopping_criteria=[stopping_criteria],
                output_hidden_states=True
            )

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
            'caption':outputs
            })
        print(image_id, outputs)

    result_file = save_result(path, args, results)

    # COCO评估
    coco = COCO(args.gt_file_path)
    coco_result = coco.loadRes(result_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()
    
    # 打印评估指标
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="llava") # llava & llava_controller & llava_verifier
    parser.add_argument("--model-vision", type=str, default="/raid_sdd/zzy/model/clip_vit_large_patch14_336")
    parser.add_argument("--bf16", action='store_true') # vision verifier needs bf16 (if train in bf16, inference need to be bf16 not fp16)
    parser.add_argument("--sigma", type=float, default=0)
    parser.add_argument("--use_verifier", action='store_true')
    parser.add_argument("--gt_file_path", type=str, default='/raid_sdd/zzy/data/halle/coco/coco2014/annotations/instances_val2014.json')
    parser.add_argument("--image_path", type=str, default='/raid_sdd/zzy/data/halle/coco/coco2014/val2014')
    parser.add_argument("--query", type=str, default="Describe this image in one sentence.")
    parser.add_argument("--conv-mode", type=str, default='v1')
    parser.add_argument("--output_folder", type=str, default='./')
    args = parser.parse_args()
    eval_model(args)