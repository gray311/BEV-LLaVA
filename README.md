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

