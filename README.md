# BEV-LLaVA


## Install

1. Clone this repository and navigate to BEV LLaVA folder

```
git clone https://github.com/gray311/BEV-LLaVA.git
cd BEV-LLaVA
```

2. Install Package
```
conda create -n bev-llava python=3.8 -y
conda activate bev-llava
pip install --upgrade pip
pip install -e .
```

3. Install Package Related to BEVFormer
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
mim install "mmdet3d>=1.1.0rc0"
```

4. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Data Preparation
1. prepare Nuscene Dataset: Please refer to [this process](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md)


2. download the DriveLM dataset:
```
pip install gdown
(1) v1_0_train_nus_refined.json (380k QA pairs): gdown --id 1HO0xVRu8awfSha_TgSeRnc2c589AjtUd
(2) v1_0_train_nus_filtered.json (70k QA pairs): gdown --id 1KtCHbf3MV7WUR08AjWzw1DuM8A4xnPqk
(3) v1_0_train_nus_mini.json (18k QA pairs): gdown --id 1Isc41fp9b0T0rtXKBXhjeYxoma6lPbMm
```


**Folder structure**
```
workspace
├── data/
|   |── drivelm/
|   |   |── v1_0_train_nus_refined.json
|   |   |── v1_0_train_nus_filtered.json
|   |   |── v1_0_train_nus_mini.json
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
```

## Model Weigths

1. download bevformer weights  (BEVFormer-base 24ep R101-DCN), you can also download the weights from [its repo](https://github.com/fundamentalvision/BEVFormer?tab=readme-ov-file).
```
cd BEV-LLaVA/llava/model/multimodal_encoder/bev_mmdet3d/ckpts/
wget https://objects.githubusercontent.com/github-production-release-asset-2e65be/501548489/d90e1b1f-0b3c-41a8-beec-21c24941adb1?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240101%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240101T144406Z&X-Amz-Expires=300&X-Amz-Signature=ac985a26b5e4f21db33c09d373b8bfc6c71e54b36dc08e3ece4416af1ee98233&X-Amz-SignedHeaders=host&actor_id=64787301&key_id=0&repo_id=501548489&response-content-disposition=attachment%3B%20filename%3Dbevformer_r101_dcn_24ep.pth&response-content-type=application%2Foctet-stream
```

2. download llava weights:
```
(1) [mm_projection weights](https://github.com/gray311/BEV-LLaVA/tree/main/workspace/checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5).
(2) [LLaVA-v1.5-vicuna-7b weights](https://huggingface.co/liuhaotian/llava-v1.5-7b)
```

## Inference

1. Inference for map image
```
python inference4img.py
```

2. Inference for bev image
```
python inference4bev.py
```

## Training

1. Training for bev image
```
bash scripts/bev_llava/finetune_bev.sh
```

2. Training for map image
```
bash scripts/bev_llava/finetune_img.sh
```

