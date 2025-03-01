import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import requests
from io import BytesIO

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def eval_single_example(args):
    # Model Initialization
    model_path = args.model_path
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, args.model_version, args.model_vision, load_bf16=args.bf16
    )
    
    if args.model_version == 'llava_controller':
        model.sigma = args.sigma
    elif args.model_version == 'llava_verifier':
        if not args.use_verifier:
            model.alpha = torch.nn.Parameter(torch.tensor(0.0))
    model = model.cuda()
    
    # Prepare the query
    qs = args.query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    print("Query: ", qs)
    
    # Set up the conversation
    conv_mode = args.conv_mode
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Load the single image
    image = load_image(args.image_file)
    if args.bf16:
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(torch.bfloat16).cuda()
    else:
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    
    # Tokenize the input
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    # Set up stopping criteria
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    # Generate the output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_length=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )
    
    # Decode the output
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    
    print("Generated Caption: ", outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="llava")
    parser.add_argument("--model-vision", type=str, default="/raid_sdd/zzy/model/clip_vit_large_patch14_336")
    parser.add_argument("--bf16", action='store_true')
    parser.add_argument("--sigma", type=float, default=0)
    parser.add_argument("--use_verifier", action='store_true')
    parser.add_argument("--image-file", type=str, required=True, help="Path to the single image file")
    parser.add_argument("--query", type=str, default="Describe this image as detailed as possible.")
    parser.add_argument("--conv-mode", type=str, default='v1')
    args = parser.parse_args()
    eval_single_example(args)