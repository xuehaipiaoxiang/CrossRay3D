# Environment Setup

## Base Environments  
Python >= 3.8 \
CUDA == 11.2 \
PyTorch == 1.11.0+cu113 \
mmdet3d == 1.0.0rc6 \
[flash-attn](https://github.com/HazyResearch/flash-attention) == 0.2.2

**Notes**: 
- [flash-attn](https://github.com/HazyResearch/flash-attention) is an optional requirement, which can speedup and save GPU memory. If your device (e.g. TESLA V100) is not compatible with the flash-attn, you can skip its installation and comment the relevant [code](../projects/mmdet3d_plugin/models/utils).
- It is also possible to consider installing version 1.9.0 + of Pytorch, but you need to find the appropriate flash-attn version (e.g. 0.2.8 for CUDA 11.6 and pytorch 1.13).


## Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation


**a. Create a conda virtual environment and activate it.**
```shell
conda create -n cr3d python=3.8 -y
conda activate cr3d
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

**c. Install flash-attn (optional).**
```
pip install flash-attn==0.2.2
```

**d. Clone CR3D.**


**e. Install mmdet3d.**
```shell
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6 
pip install -e .
```

**f. Train CR3D in fusion manner.**
```
python ./bash/dist_train.sh ./project/sparse_image.py 2 # firstly 
python ./bash/dist_train.sh ./project/sparse_lidar.py 2 # then
merge  weight from image and lidar
python ./bash/dist_train.sh ./project/sparse_fusion.py 2 # finally
```
