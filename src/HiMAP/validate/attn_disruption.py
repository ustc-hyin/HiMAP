import argparse
import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from functools import partial
from PIL import Image
import math

from disruption_forward import disruption_forward

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--isolate-modality", type=str)
    parser.add_argument("--isolate-layer", type=int)
    args = parser.parse_args()

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    model.config.use_hmap_v = False
    model.model.reset_hmapv()

    # disrupte the information flow
    model.model.forward = \
        partial(disruption_forward, model.model, isolate_modality=args.isolate_modality, isolate_layer=args.isolate_layer)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    label_consistency = 0
    score_consistency = 0

    for i, line in enumerate(tqdm(questions)):

        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs
        label = line['conversations'][1]['value']
        top5_ref_opt = line['top5_res']

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            continue

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                max_new_tokens=8,
                use_cache=False,
                stopping_criteria=stopping_criteria,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids['sequences'][:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids['sequences'][:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        if label in outputs:
            label_consistency += 1

        top5_res_opt = output_ids['scores'][0].topk(5).indices[0].tolist()        

    #     # compute jaccard similarity
        set_res = set(top5_res_opt)
        set_ref = set(top5_ref_opt)
        intersection = set_res.intersection(set_ref)
        union = set_res.union(set_ref)
        jaccard = len(intersection) / len(union)
        score_consistency += jaccard

    num_sample = len(questions)
    
    print('isolate_modality:', args.isolate_modality, 'isolate_layer', args.isolate_layer)
    print('label_consistency', label_consistency/num_sample, 'score_consistency', score_consistency/num_sample)

    