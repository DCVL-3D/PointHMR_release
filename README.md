# PointHMR

This repository is a official Pytorch implementation of the paper [**"Sampling is Matter: Point-guided 3D Human Mesh Reconstruction"**](https://) <br>
Jeonghwan Kim*, Mi-Gyeong Gwon*, Hyunwoo Park, Hyukmin Kwon, Gi-Mun Um, and Wonjun Kim (Corresponding Author) <br>
\* equally contributed <br>
:maple_leaf: ***IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)***, Jun. 2023. :maple_leaf:

## Overview :eyes:
- We propose to utilize the correspondence between encoded features and vertex positions, which are projected into the 2D space, via our point-guided feature sampling scheme. By explicitly indicating such vertex-relevant features to the transformer encoder, coordinates of the 3D human mesh are accurately estimated.
- Our progressive attention masking scheme helps the model efficiently deal with local vertex-to-vertex relations even under complicated poses and occlusions.
    <p align="center"><img src='https://github.com/DCVL-3D/PointHMR_release/blob/main/documents/fig1.png'></p>


## How to use it

### Try on Google Colab
It allows you to run the project in the cloud, free of charge. 
Let's give the prepared [Google Colab demo](https://colab.research.google.com/) a try.

### Installation

Please refer to [install.md](documents/installation.md) for installation.

### Demo

Currently, we support processing images, video or real-time webcam.    
Pelease refer to [config_guide.md](docs/config_guide.md) for configurations.   
ROMP can be called as a python lib inside the python code, jupyter notebook, or from command line / scripts, please refer to [Google Colab demo](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg) for examples.

<!-- ## Installation
## Evaluation
## Demo
## Contribution -->
