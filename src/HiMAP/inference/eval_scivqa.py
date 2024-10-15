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

from PIL import Image
import math

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
    # HiMAP hyperparameter
    parser.add_argument('--use-hmap-v', default=False, action='store_true', help='whether to use hmap-v')
    parser.add_argument('--sys-length', type=int, required=False, help='the length of system prompt')
    parser.add_argument('--img-length', type=int, required=False, help='the length of image token')
    parser.add_argument('--hmap-v-attn-txt-layer', type=int, required=False, help='the layer of pruning accorading to img2txt information')
    parser.add_argument('--hmap-v-attn-img-layer', type=int, required=False, help='the layer of pruning accorading to img2img information')
    parser.add_argument('--hmap-v-attn-txt-rank', type=int, required=False, help='the rank of attn accorading to img2txt information')
    parser.add_argument('--hmap-v-attn-img-rank', type=int, required=False, help='the rank of attn accorading to img2img information')
    args = parser.parse_args()

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # set model hmapv config
    if args.use_hmap_v == True:
        model.config.use_hmap_v = True
        model.config.hmap_v_sys_length = args.sys_length
        model.config.hmap_v_img_length = args.img_length
        model.config.hmap_v_attn_txt_layer = args.hmap_v_attn_txt_layer
        model.config.hmap_v_attn_img_layer = args.hmap_v_attn_img_layer
        model.config.hmap_v_attn_txt_rank = args.hmap_v_attn_txt_rank
        model.config.hmap_v_attn_img_rank = args.hmap_v_attn_img_rank
        print('HiMAP TECHNIQUE WILL BE USED ------')      
    else:
        model.config.use_hmap_v = False
        print('NO TOKEN PRUNING TCHNIQUE WILL BE USED ------')

    model.model.reset_hmapv()

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    num_sample = len(questions)
    corr_sample = 0

    for i, line in enumerate(tqdm(questions)):
        
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs
        label = line['conversations'][1]['value']

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

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                max_new_tokens=1024,
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

        if outputs == label:
            corr_sample += 1
    
    print(corr_sample/num_sample)