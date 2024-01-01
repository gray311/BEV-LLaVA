import os
import torch
from llava.model import *
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.model.multimodal_encoder.bev_mmdet3d.datasets import (
    custom_build_dataset,
    CustomNuScenesDataset,
)
from mmengine import Config

def load_bev_model(model_base, checkpoint_path, load_8bit, load_4bit, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)

    config = AutoConfig.from_pretrained(checkpoint_path)
    config.pretrain_mm_mlp_adapter = None
    model.get_model().initialize_bev_modules(model_args=config)
    bev_tower = model.get_bev_tower()
    bev_tower.to(dtype=torch.float16, device=device) # Currently bf16 is not supported because of bevformer

    mm_projector_weights = torch.load(os.path.join(checkpoint_path, 'mm_projector.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    model.load_state_dict(mm_projector_weights, strict=False)
    context_len = 2048
    model.to(device).half()
    return tokenizer, model, context_len

def main():
    disable_torch_init()
    model_path = 'liuhaotian/llava-v1.5-7b'
    checkpoint_path = '/home/scratch.chaoweix_nvresearch/visual_instruction/BEV-LLaVA/workspace/checkpoints/bev-vicuna-v1.0-7b-pretrain'
    device = 'cuda:0'
    load_4bit, load_8bit = False, False
    tokenizer, model, context_len = load_bev_model(model_path, checkpoint_path, load_8bit, load_4bit, device=device)

    data_config = "./llava/model/multimodal_encoder/bev_mmdet3d/configs/bevformer.py"
    cfg = Config.fromfile(data_config)
    # cfg.data.test.test_mode = True
    list_data_dict = custom_build_dataset(cfg.data.test)


    from tqdm import tqdm
    for i in range(len(list_data_dict)):
        source = list_data_dict[i]
        sample_token = source['sample_token']
        source['conversations'] = [
            {
                "from": "human",
                "value": source['instruction'].format(question=source['question'])
            },
            {
                "from": "gpt",
                "value": source['answer']
            },

        ]

        instruction = source['conversations'][0]['value']
        instruction = instruction.replace(DEFAULT_X_TOKEN['BEV'], '').strip()
        instruction = DEFAULT_X_TOKEN['BEV'] + '\n' + instruction
        instruction = instruction.strip()

        bev_img = source['img'].unsqueeze(0).to(device).half()
        bev_img_metas = {}
        for j, img_meta in source['img_metas'].items():
            bev_img_metas[j] = {key: value for key, value in img_meta.items() \
                            if key in ['scene_token', 'prev_bev_exists', 'img_shape', 'can_bus', 'lidar2img']}
        bev_img_metas = [bev_img_metas]
        # image_processor = processor['image']
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        key = ['bev']

        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_X_token(prompt, tokenizer, [X_TOKEN_INDEX['BEV']], return_tensors='pt').unsqueeze(0).to(device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                img=bev_img,
                img_metas=bev_img_metas,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        print(f"========={i}: {sample_token}=========")
        print(f"instruction: {instruction}")
        print(f"prediction: {outputs}")
        print(f"reference: {source['conversations'][1]['value']}")



if __name__ == '__main__':
    main()

    # checkpoint_path = '/home/scratch.chaoweix_nvresearch/visual_instruction/BEV-LLaVA/workspace/checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/'
    # mm_projector_weights = torch.load(os.path.join(checkpoint_path, 'mm_projector.bin'), map_location='cpu')
    # mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    # for k, v in mm_projector_weights.items():
    #     print(k, v.shape)
