# PointHMR

This repository is a Pytorch implementation of the paper [**"Sampling is Matter: Point-guided 3D Human Mesh Reconstruction"**](https://)
Jeonghwan Kim, Mi-Gyeong Gwon, Hyunwoo Park, Hyukmin Kwon, Gi-Mun Um, and Wonjun Kim*
IEEE Conference on Computer Vision and Pattern Recognition 2023

## Overview
- We propose to utilize the correspondence between encoded features and vertex positions, which are projected into the 2D space, via our point-guided feature sampling scheme. By explicitly indicating such vertex-relevant features to the transformer encoder, coordinates of the 3D human mesh are accurately estimated.
- Our progressive attention masking scheme helps the model efficiently deal with local vertex-to-vertex relations even under complicated poses and occlusions.
    <p align="center"><img src='https://github.com/DCVL-3D/PointHMR_release/blob/main/documents/fig1.png'></p>

    
