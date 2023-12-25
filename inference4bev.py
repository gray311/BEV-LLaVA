import torch

print(torch.version.cuda)


model = torch.load("/home/scratch.chaoweix_nvresearch/visual_instruction/BEV-LLaVA/workspace/checkpoints/bev-llava-v1.5-7b-pretrain/checkpoint-10/mm_projector.bin")

print(model.keys())