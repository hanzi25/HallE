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
import matplotlib.pyplot as plt

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
    if args.model_version == 'llava_verifier':
        if not args.use_verifier:
            model.alpha = 0.0
        else:
            model.alpha = 1.0
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
    
    # Generate the verified output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_length=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # import pdb; pdb.set_trace()

    # Decode the output
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids["sequences"][:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]

    verified_logits = output_ids["scores"]

    print("Generated Caption:", outputs)
    print()

    # Generate the original output
    with torch.inference_mode():
        model.alpha = 0.0
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_length=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Decode the output
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids["sequences"][:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]

    original_logits = output_ids["scores"]

    print("Original Caption:", outputs)

    # import pdb; pdb.set_trace()

    # Token ranking
    token_str = "door"
    token_id = tokenizer.encode(token_str, add_special_tokens=False)[0]
    original_token_ranks, verified_token_ranks = get_token_rankings(original_logits, verified_logits, token_id)
    plot_token_rankings(token_str, original_token_ranks, verified_token_ranks)


def get_token_rankings(original_logits, verified_logits, target_token_id):
    original_seq_len = len(original_logits)
    original_vocab_size = original_logits[0].shape[-1]

    original_token_ranks = []
    for t in range(original_seq_len):
        original_logits_t = original_logits[t]
        original_rankings_t = torch.argsort(original_logits_t, dim=-1, descending=True)  # shape: (1, vocab_size)
        original_rank = (original_rankings_t[0] == target_token_id).nonzero(as_tuple=False)
        if len(original_rank) > 0:
            original_token_ranks.append(original_rank.item())
        else:
            original_token_ranks.append(None)

    verified_seq_len = len(verified_logits)
    verified_vocab_size = verified_logits[0].shape[-1]

    verified_token_ranks = []
    for t in range(verified_seq_len):
        verified_logits_t = verified_logits[t]
        verified_rankings_t = torch.argsort(verified_logits_t, dim=-1, descending=True)
        verified_rank = (verified_rankings_t[0] == target_token_id).nonzero(as_tuple=False)
        if len(verified_rank) > 0:
            verified_token_ranks.append(verified_rank.item())
        else:
            verified_token_ranks.append(None)
    
    return original_token_ranks, verified_token_ranks

def plot_token_rankings(token_str, original_token_ranks, verified_token_ranks):
    original_steps = list(range(len(original_token_ranks)))
    original_ranks = [r + 1 if r >= 0 else 32000 for r in original_token_ranks] # in case of log(0)
    verified_steps = list(range(len(verified_token_ranks)))
    verified_ranks = [r + 1 if r >= 0 else 32000 for r in verified_token_ranks]

    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1)
    plt.plot(original_steps, original_ranks, linestyle='-')
    plt.gca().invert_yaxis()  # 排名越高（数字越小）越靠上

    plt.yscale('log')  # 设置 log 纵轴（对数坐标）
    plt.yticks([1, 10, 100, 1000, 10000, 32000], labels=["1", "10", "100", "1k", "10k", "32k"])
    plt.xlabel("Time Step")
    plt.ylabel("Ranking of Token")
    plt.title(f"Ranking of token '{token_str}' before revision over time")
    plt.grid(True)

    # 添加文字标注
    for x, y in zip(original_steps, original_ranks):
        if y <= 100:  # 如果不是 None
            plt.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 5),
                         ha='center', fontsize=8, color='blue')
    
    plt.subplot(2,1,2)
    plt.plot(verified_steps, verified_ranks, linestyle='-')
    plt.gca().invert_yaxis()  # 排名越高（数字越小）越靠上

    plt.yscale('log')  # 设置 log 纵轴（对数坐标）
    plt.yticks([1, 10, 100, 1000, 10000, 32000], labels=["1", "10", "100", "1k", "10k", "32k"])
    plt.xlabel("Time Step")
    plt.ylabel("Ranking of Token")
    plt.title(f"Ranking of token '{token_str}' after revision over time")
    plt.grid(True)

    # 添加文字标注
    for x, y in zip(verified_steps, verified_ranks):
        if y <= 100:  # 如果不是 None
            plt.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 5),
                         ha='center', fontsize=8, color='blue')
    plt.tight_layout()
    plt.savefig("token_rank.png", dpi=300)


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="llava")
    parser.add_argument("--model-vision", type=str, default="/raid_sdd/zzy/model/clip_vit_large_patch14_336")
    parser.add_argument("--bf16", action='store_true')
    parser.add_argument("--use_verifier", action='store_true')
    parser.add_argument("--image-file", type=str, required=True, help="Path to the single image file")
    parser.add_argument("--query", type=str, default="Describe this image as detailed as possible.")
    parser.add_argument("--conv-mode", type=str, default='v1')
    args = parser.parse_args()
    eval_single_example(args)