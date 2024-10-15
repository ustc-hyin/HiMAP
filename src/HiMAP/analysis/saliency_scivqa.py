import argparse
import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image
import math

from atten_adapter import AttnAdapter, get_props
import torch.nn.functional as F

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
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    model.config.use_hmap_v = False
    model.model.reset_hmapv()

    # Change the attention module 
    for layer in model.model.layers:
        attn_adap = AttnAdapter(layer.self_attn.config)
        attn_adap.load_state_dict(layer.self_attn.state_dict())
        attn_adap = attn_adap.half().cuda()
        layer.self_attn = attn_adap

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    proportions = []

    for i, line in enumerate(tqdm(questions)):

        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        # get the true option
        answer = line['conversations'][1]['value']
        label = tokenizer.convert_tokens_to_ids(answer)
        label = torch.tensor(label, dtype=torch.int64)
        label = label.cuda()

        image_file = line["image"]
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        images = image_tensor.unsqueeze(0).half().cuda()
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        cur_prompt = '<image>' + '\n' + cur_prompt


        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        model.zero_grad()
        output_ids = model.forward(
            input_ids,
            images=images,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

        pred_logit = output_ids['logits'][:,-1,:].squeeze(0)
        loss = F.cross_entropy(pred_logit, label)
        loss.backward()

        # compute the saliency score
        props_all, props_img = [], []
        for idx_layer, layer in enumerate(model.model.layers):

            attn_grad = layer.self_attn.attn_map.grad.detach().clone().cpu()
            attn_map = output_ids['attentions'][idx_layer].detach().clone().cpu()
            saliency = torch.abs(attn_grad * attn_map)

            props4all, props4img = get_props(saliency)
            props_all.append(props4all)
            props_img.append(props4img)

        proportions.append((props_all, props_img))
    
    torch.save(proportions, args.result_file)

            



