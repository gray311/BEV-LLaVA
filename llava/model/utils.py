from transformers import AutoConfig
import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)


def convert_color(img_path):
    plt.figure()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.imsave(img_path, img, cmap=plt.get_cmap('viridis'))
    plt.close()


def save_tensor(tensor, path, pad_value=254.0,):
    print('save_tensor', path)
    tensor = tensor.to(torch.float).detach().cpu()
    if tensor.type() == 'torch.BoolTensor':
        tensor = tensor*255
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    tensor = make_grid(tensor, pad_value=pad_value, normalize=False).permute(1, 2, 0).numpy().copy()
    torchvision.utils.save_image(torch.tensor(tensor).permute(2, 0, 1), path)
    convert_color(path)


def plot_attention_map(tensor, path):
    print('plot attention map', path)
    if len(tensor.shape) == 3:
        tensor = torch.sum(tensor, dim=-1)
    assert len(tensor.shape) == 2
    tensor = tensor.to(torch.float).detach().cpu()

    import cv2
    import numpy as np

    tensor = cv2.resize(tensor.numpy(), (200, 200))
    tensor = cv2.GaussianBlur(tensor, (11, 11), sigmaX=0, sigmaY=0)

    # Normalizing the smoothed tensor
    tensor = tensor / tensor.max()
    tensor = (tensor * 255).astype('uint8')

    # Displaying the normalized and smoothed tensor with 'viridis' colormap and without interpolation for a smoother effect
    plt.figure(figsize=(10, 10))
    plt.imshow(tensor, alpha=1.0, interpolation='nearest', cmap='viridis')
    plt.savefig(path)

