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
```

4. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
