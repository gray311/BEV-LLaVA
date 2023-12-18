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
bash scripts/bev_llava/stage_1.sh
```

2. Training for map image
```
in progress
```

