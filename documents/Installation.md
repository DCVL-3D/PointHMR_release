## Installation

Our codebase is developed based on Ubuntu 18.04 and Pytorch framework.

### Requirements

* Python >=3.8
* Pytorch >=1.7.1
* torchvision >= 0.8.2
* CUDA >= 10.1
* cuDNN (if CUDA available)

### Installation with conda

git clone https://github.com/DCVL-3D/PointHMR_release
cd PointHMR_release

```bash
# We suggest to create a new conda environment with python version 3.8
conda create --name PHMR python=3.8

# Activate conda environment
conda activate PHMR

# Install Pytorch that is compatible with your CUDA version
# CUDA 10.1
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
# CUDA 10.2
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
# CUDA 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# Install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# Install OpenDR
pip install matplotlib
pip install git+https://gitlab.eecs.umich.edu/ngv-python-modules/opendr.git

# Install PointHMR
git clone --recursive https://github.com/DCVL-3D/PointHMR_release
cd PointHMR_release
python setup.py build develop

# Install requirements
pip install -r requirements.txt
pip install ./manopth/.


```
