import torch
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image

def main():
    disable_torch_init()
    image = '/home/scratch.chaoweix_nvresearch/visual_instruction/BEV-LLaVA/images/driving_1.jpg'
    instruction = 'What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.'
    model_path = '/home/scratch.chaoweix_nvresearch/visual_instruction/BEV-LLaVA/workspace/checkpoints/img-vicuna-v1.0-7b-pretrain'
    model_base = 'lmsys/vicuna-7b-v1.5'
    device = 'cuda'
    load_4bit, load_8bit = False, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, "llava", load_8bit, load_4bit, device=device, driving_scene="map")
    image_processor = processor['image']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    image_tensor = image_processor.preprocess(Image.open(image).convert("RGB"), return_tensors='pt')['pixel_values']
    if type(image_tensor) is list:
        tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        tensor = image_tensor.to(model.device, dtype=torch.float16)
    key = ['image']

    print(f"{roles[1]}: {instruction}")
    instruction = DEFAULT_X_TOKEN['IMAGE'] + '\n' + instruction
    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_X_token(prompt, tokenizer, [X_TOKEN_INDEX['IMAGE']], return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)

if __name__ == '__main__':
    main()