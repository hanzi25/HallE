import argparse
import torch
from tqdm import tqdm
import os
import sys
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import pickle

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import requests
from PIL import Image
from io import BytesIO
import json
import random
import datetime

# copied from: https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
synonyms_txt = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison 
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield 
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''

def prepare_nlp(args):
    # ========================================
    #            Prepare NLP
    # ========================================
    imid_to_objects = defaultdict(list)
    coco_path = args.annotation_path
    #read in synonyms
    synonyms = synonyms_txt.splitlines()
    synonyms = [s.strip().split(', ') for s in synonyms]
    mscoco_objects = [] #mscoco objects and *all* synonyms
    inverse_synonym_dict = {}
    for synonym in synonyms:
        mscoco_objects.extend(synonym)
        for s in synonym:
            inverse_synonym_dict[s] = synonym[0]

    #Some hard coded rules for implementing CHAIR metrics on MSCOCO
    
    #common 'double words' in MSCOCO that should be treated as a single word
    coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']
    
    #Hard code some rules for special cases in MSCOCO
    #qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
    animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
    #qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
    vehicle_words = ['jet', 'train']
    
    #double_word_dict will map double words to the word they should be treated as in our analysis
    
    double_word_dict = {}
    for double_word in coco_double_words:
        double_word_dict[double_word] = double_word
    for animal_word in animal_words:
        double_word_dict['baby %s' %animal_word] = animal_word
        double_word_dict['adult %s' %animal_word] = animal_word
    for vehicle_word in vehicle_words:
        double_word_dict['passenger %s' %vehicle_word] = vehicle_word
    double_word_dict['bow tie'] = 'tie'
    double_word_dict['toilet seat'] = 'toilet'
    double_word_dict['wine glas'] = 'wine glass'

    imid_to_objects = get_annotations(coco_path, imid_to_objects, double_word_dict, mscoco_objects, inverse_synonym_dict)
    
    return coco_path, imid_to_objects, double_word_dict, mscoco_objects, inverse_synonym_dict


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def save_result(path, args, results):

    if args.model_version == 'llava_controller':
        save_file = f"{path}/rank_score_{args.model_version}_{args.sigma}.jsonl"
    elif args.model_version == 'llava_verifier':
        if not args.use_verifier:
            save_file = f"{path}/rank_score_{args.model_version}_no_verifier.jsonl"
        else:
            save_file = f"{path}/rank_score_{args.model_version}.jsonl"
    else:
        save_file = f"{path}/rank_score_{args.model_version}.jsonl"

    transformed_results = []
    metrics_sums = {
        'rank_score_gt_o': 0,
        'rank_score_gt_v': 0,
        'rank_score_hal_o': 0,
        'rank_score_hal_v': 0,
    }
    num_items = len(results)

    for item in results:
        metrics = {
            'mscoco_grounded_words_o': item['mscoco_grounded_words_o'],
            'grounded_words_count_o': item['grounded_words_count_o'],
            'mscoco_omitted_words_o': item['mscoco_omitted_words_o'],
            'omitted_words_count_o': item['omitted_words_count_o'],
            'mscoco_hallucinated_words_o': item['mscoco_hallucinated_words_o'],
            'hallucinated_words_count_o': item['hallucinated_words_count_o'],
            'mscoco_grounded_words_v': item['mscoco_grounded_words_v'],
            'grounded_words_count_v': item['grounded_words_count_v'],
            'mscoco_omitted_words_v': item['mscoco_omitted_words_v'],
            'omitted_words_count_v': item['omitted_words_count_v'],
            'mscoco_hallucinated_words_v': item['mscoco_hallucinated_words_v'],
            'hallucinated_words_count_v': item['hallucinated_words_count_v'],
            'rank_score_gt_o': item['rank_score_gt_o'],
            'rank_score_gt_v': item['rank_score_gt_v'],
            'rank_score_hal_o': item['rank_score_hal_o'],
            'rank_score_hal_v': item['rank_score_hal_v'],
        }

        for k in metrics_sums:
            metrics_sums[k] += metrics[k]

        transformed_results.append({
            "sentences": {
                "image_id": item['image_id'],
                "image_file": item['image_file'],
                "verified_caption": item['verified_caption'],
                "original_caption": item['original_caption'],
                "metrics": metrics
            }
        })

    overall_metrics = {
        k.replace('score', 'avg_score'): round(v / num_items, 4) for k, v in metrics_sums.items()
    }

    final_output = {
        "results": transformed_results,
        "overall_metrics": overall_metrics
    }

    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)


def combine_coco_instances(annotation_path):
    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO instance annotations for val set")
    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO instance annotations for train set")

    val_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'val')))
    train_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'train')))
    all_instances = {'info': train_instances['info'],
                     'licenses': train_instances['licenses'],
                     'type': train_instances['licenses'],
                     'categories': train_instances['categories'],
                     'images': train_instances['images'] + val_instances['images'],
                     'annotations': val_instances['annotations'] + train_instances['annotations']}

    return all_instances


def combine_coco_captions(annotation_path):
    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO caption annotations for val set")
    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'val')))
    train_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'train')))
    all_caps = {'info': train_caps['info'],
                'licenses': train_caps['licenses'],
                'images': val_caps['images'] + train_caps['images'],
                'annotations': val_caps['annotations'] + train_caps['annotations']}

    return all_caps 


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def caption_to_words(caption, double_word_dict, mscoco_objects, inverse_synonym_dict):
    '''
    Input: caption
    Output: MSCOCO words in the caption
    '''

    #standard preprocessing
    words = nltk.word_tokenize(caption.lower())
    tagged_sent = nltk.pos_tag(words)
    lemmas_sent = []
    wnl = WordNetLemmatizer()
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    # words = [singularize(w) for w in words]
    words = lemmas_sent

    #replace double words
    i = 0
    double_words = []
    idxs = []
    while i < len(words):
        idxs.append(i) 
        double_word = ' '.join(words[i:i+2])
        if double_word in double_word_dict: 
            double_words.append(double_word_dict[double_word])
            i += 2
        else:
            double_words.append(words[i])
            i += 1
    words = double_words

    #toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
    if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']

    #get synonyms for all words in the caption
    idxs = [idxs[idx] for idx, word in enumerate(words) \
            if word in set(mscoco_objects)]
    words = [word for word in words if word in set(mscoco_objects)]
    node_words = []
    for word in words:
        node_words.append(inverse_synonym_dict[word])
    #return all the MSCOCO objects in the caption
    return words, node_words, idxs, double_words


def get_annotations_from_segments(coco_path, imid_to_objects, inverse_synonym_dict):
    '''
    Add objects taken from MSCOCO segmentation masks
    '''

    coco_segments = combine_coco_instances(coco_path)
    segment_annotations = coco_segments['annotations']

    #make dict linking object name to ids
    id_to_name = {} #dict with id to synsets 
    for cat in coco_segments['categories']:
        id_to_name[cat['id']] = cat['name']

    for i, annotation in enumerate(segment_annotations):
        sys.stdout.write("\rGetting annotations for %d/%d segmentation masks" 
                            %(i, len(segment_annotations)))
        imid = annotation['image_id']
        
        node_word = inverse_synonym_dict[id_to_name[annotation['category_id']]]
        imid_to_objects[imid].append(node_word)
    print("\n")
    
    return imid_to_objects


def get_annotations_from_captions(coco_path, imid_to_objects, double_word_dict, mscoco_objects, inverse_synonym_dict):
    '''
    Add objects taken from MSCOCO ground truth captions 
    '''

    coco_caps = combine_coco_captions(coco_path)
    caption_annotations = coco_caps['annotations']

    for i, annotation in enumerate(caption_annotations):
        sys.stdout.write('\rGetting annotations for %d/%d ground truth captions' 
                            %(i, len(coco_caps['annotations'])))
        imid = annotation['image_id']
        
        _, node_words, _, _ = caption_to_words(annotation['caption'], double_word_dict, mscoco_objects, inverse_synonym_dict)
        # note here is update, so call get_annotations_from_segments first
        imid_to_objects[imid].extend(node_words)
    print("\n")

    return imid_to_objects


def get_annotations(coco_path, imid_to_objects, double_word_dict, mscoco_objects, inverse_synonym_dict):

    '''
    Get annotations from both segmentation and captions.  Need both annotation types for CHAIR metric.
    '''
    
    imid_to_objects = get_annotations_from_segments(coco_path, imid_to_objects, inverse_synonym_dict) 
    imid_to_objects = get_annotations_from_captions(coco_path, imid_to_objects, double_word_dict, mscoco_objects, inverse_synonym_dict)

    # deduplicate
    for imid in imid_to_objects:
        imid_to_objects[imid] = set(imid_to_objects[imid])

    return imid_to_objects


def generate_wordlist(imid_to_objects, imid, cap, double_word_dict, mscoco_objects, inverse_synonym_dict):
    
    grounded_words = set()
    omitted_words = set()
    hallucinated_words = set()

    words, node_words, idxs, raw_words = caption_to_words(cap, double_word_dict, mscoco_objects, inverse_synonym_dict)
    gt_objects = imid_to_objects[imid]
    gt_words = list(gt_objects)
    generated_words = list(node_words)

    for word, node_word, idx in zip(words, node_words, idxs):
        if node_word not in gt_objects:
            hallucinated_words.add(node_word)
        else:
            grounded_words.add(node_word)
    
    for gt_object in gt_objects:
        if gt_object not in node_words:
            omitted_words.add(gt_object)
    
    return grounded_words, omitted_words, hallucinated_words, gt_words, generated_words, raw_words


def compute_rank_score(words, tokenizer, logits):
    seq_len = len(logits)
    vocab_size = logits[0].shape[-1]

    rank_score = 0.
    for w in words:
        token_id = tokenizer.encode(w, add_special_tokens=False)[0]
        token_ranks = []
        for t in range(seq_len):
            logits_t = logits[t]
            rankings_t = torch.argsort(logits_t, dim=-1, descending=True)
            rank = (rankings_t[0] == token_id).nonzero(as_tuple=False)
            if len(rank) > 0:
                token_ranks.append(rank.item())
            else: token_ranks.append(None)
        
        ranks = [r + 1 if r >= 0 else 32000 for r in token_ranks]
        for r in ranks:
            if r <= 100:
                rank_score += 1 / r

    return rank_score


def compute_metrics(args, coco_path, imid_to_objects, double_word_dict, mscoco_objects, inverse_synonym_dict, tokenizer, image_id, verified_outputs, verified_logits, original_outputs, original_logits):


    grounded_words_o, omitted_words_o, hallucinated_words_o, gt_words, generated_words_o, raw_words_o = generate_wordlist(imid_to_objects, image_id, original_outputs, double_word_dict, mscoco_objects, inverse_synonym_dict)
    grounded_words_v, omitted_words_v, hallucinated_words_v, _ , generated_words_v, raw_words_v = generate_wordlist(imid_to_objects, image_id, verified_outputs, double_word_dict, mscoco_objects, inverse_synonym_dict)

    hallucinated_words = list(hallucinated_words_o | hallucinated_words_v)

    rank_score_gt_o = compute_rank_score(gt_words, tokenizer, original_logits)
    rank_score_hal_o = compute_rank_score(hallucinated_words, tokenizer, original_logits)
    rank_score_gt_v = compute_rank_score(gt_words, tokenizer, verified_logits)
    rank_score_hal_v = compute_rank_score(hallucinated_words, tokenizer, verified_logits)

    return list(grounded_words_o), list(omitted_words_o), list(hallucinated_words_o), list(grounded_words_v), list(omitted_words_v), list(hallucinated_words_v), rank_score_gt_o, rank_score_hal_o, rank_score_gt_v, rank_score_hal_v


def eval_model(args, coco_path, imid_to_objects, double_word_dict, mscoco_objects, inverse_synonym_dict):

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

        # Load the single image
        image = load_image(image_file)
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

            # import pdb; pdb.set_trace()
            # model.config.output_attentions = True  # 启用注意力输出
            # generated_ids, all_attention_scores = custom_generate_with_attention(model, input_ids, image_tensor, max_new_tokens=512)
            # import pdb; pdb.set_trace()

            verified_output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_length=1024,
                # max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                return_dict_in_generate=True,
                output_scores=True
            )

        # Decode the output
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != verified_output_ids["sequences"][:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        verified_outputs = tokenizer.batch_decode(verified_output_ids["sequences"][:, input_token_len:], skip_special_tokens=True)[0]
        verified_outputs = verified_outputs.strip()
        if verified_outputs.endswith(stop_str):
            verified_outputs = verified_outputs[:-len(stop_str)]

        verified_logits = verified_output_ids["scores"]

        verified_outputs = verified_outputs.strip()

        print(image_id)

        print("Verified Caption:", verified_outputs, "\n")

        # Generate the original output
        with torch.inference_mode():
            model.alpha = 0.0
            original_output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_length=1024,
                # max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode the output
        n_diff_input_output = (input_ids != original_output_ids["sequences"][:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        original_outputs = tokenizer.batch_decode(original_output_ids["sequences"][:, input_token_len:], skip_special_tokens=True)[0]
        original_outputs = original_outputs.strip()
        if original_outputs.endswith(stop_str):
            original_outputs = original_outputs[:-len(stop_str)]

        original_logits = original_output_ids["scores"]

        original_outputs = original_outputs.strip()

        print("Original Caption:", original_outputs)

        grounded_words_o, omitted_words_o, hallucinated_words_o, grounded_words_v, omitted_words_v, hallucinated_words_v, rank_score_gt_o, rank_score_hal_o, rank_score_gt_v, rank_score_hal_v = compute_metrics(args, coco_path, imid_to_objects, double_word_dict, mscoco_objects, inverse_synonym_dict, tokenizer, image_id, verified_outputs, verified_logits, original_outputs, original_logits)

        results.append({
            'image_id': image_id,
            'image_file': image_file,
            'verified_caption': verified_outputs,
            'original_caption': original_outputs,
            'mscoco_grounded_words_o': grounded_words_o,
            'grounded_words_count_o': len(grounded_words_o),
            'mscoco_omitted_words_o': omitted_words_o,
            'omitted_words_count_o': len(omitted_words_o),
            'mscoco_hallucinated_words_o': hallucinated_words_o,
            'hallucinated_words_count_o': len(hallucinated_words_o),
            'mscoco_grounded_words_v': grounded_words_v,
            'grounded_words_count_v': len(grounded_words_v),
            'mscoco_omitted_words_v': omitted_words_v,
            'omitted_words_count_v': len(omitted_words_v),
            'mscoco_hallucinated_words_v': hallucinated_words_v,
            'hallucinated_words_count_v': len(hallucinated_words_v),
            'rank_score_gt_o': rank_score_gt_o,
            'rank_score_gt_v': rank_score_gt_v,
            'rank_score_hal_o': rank_score_hal_o,
            'rank_score_hal_v': rank_score_hal_v,
        })
        
    save_result(path, args, results)


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
    parser.add_argument("--annotation_path", type=str, default='/raid_sdd/zzy/data/halle/coco/coco2014/annotations')
    parser.add_argument("--query", type=str, default="Describe this image as detailed as possible.")
    parser.add_argument("--conv-mode", type=str, default='v1')
    parser.add_argument("--cache", type=str, default="mscoco.pkl")
    parser.add_argument("--output_folder", type=str, default='./')
    args = parser.parse_args()

    if args.cache and os.path.exists(args.cache):
        loaded_data = pickle.load(open(args.cache, 'rb'))
        print(f"loaded evaluator from cache: {args.cache}")
        coco_path, imid_to_objects, double_word_dict, mscoco_objects, inverse_synonym_dict = loaded_data
    else:
        print(f"cache not setted or not exist yet, building from scratch...")
        coco_path, imid_to_objects, double_word_dict, mscoco_objects, inverse_synonym_dict = prepare_nlp(args)
        data = (coco_path, imid_to_objects, double_word_dict, mscoco_objects, inverse_synonym_dict)
        pickle.dump(data, open(args.cache, 'wb'))
        print(f"cached evaluator to: {args.cache}")
    
    eval_model(args, coco_path, imid_to_objects, double_word_dict, mscoco_objects, inverse_synonym_dict)