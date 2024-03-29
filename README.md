<h1 align="center">Sampling is Matter: Point-guided 3D Human Mesh Reconstruction</h1>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sampling-is-matter-point-guided-3d-human-mesh-1/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=sampling-is-matter-point-guided-3d-human-mesh-1)



This repository is an official Pytorch implementation of the paper [**"Sampling is Matter: Point-guided 3D Human Mesh Reconstruction"**](https://arxiv.org/abs/2304.09502v1) <br>
Jeonghwan Kim*, Mi-Gyeong Gwon*, Hyunwoo Park, Hyukmin Kwon, Gi-Mun Um, and Wonjun Kim (Corresponding Author) <br>
\* equally contributed <br>
:maple_leaf: ***IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)***, Jun. 2023. :maple_leaf:

<p align="center"><img src='documents/fig1.jpg'></p>

## :eyes: Overview 
- We propose to utilize the correspondence between encoded features and vertex positions, which are projected into the 2D space, via our point-guided feature sampling scheme. By explicitly indicating such vertex-relevant features to the transformer encoder, coordinates of the 3D human mesh are accurately estimated.
- Our progressive attention masking scheme helps the model efficiently deal with local vertex-to-vertex relations even under complicated poses and occlusions.

<p align="center"><img src='documents/fig2.jpg'></p>


## :gear: How to use it 
<!--

_This section will be released soon!_
 ### Try on Google Colab
It allows you to run the project in the cloud, free of charge.  </br>
Let's give the prepared [Google Colab demo](https://colab.research.google.com/) a try.

 -->

### Installation

Please refer to [Installation.md](documents/Installation.md) for installation.

### Download

We provide guidelines to download pre-trained models and datasets. </br>
Please check [Download.md](documents/Download.md) for more information.

### Demo

We provide demo codes to run end-to-end inference on the test images. </br>

Please check [Demo.md](documents/Demo.md) for more information.

### Experiments

We provide guidelines to train and evaluate our model on Human3.6M and 3DPW. </br>

Please check [Experiments.md](documents/Experiments.md) for more information.


## :page_with_curl: Results 

### Quantitative result
| Model                        | Dataset   | MPJPE | PA-MPJPE | Checkpoint            |
| ---------------------------- | --------- | ----- | -------- | --------------- |
| PointHMR-HR32                | Human3.6M |48.3   | 32.9     | [Download](https://drive.google.com/file/d/1Np8SAEFEou2HcfDYH7b1a4rjLI1GnwVQ/view?usp=sharing)|
| PointHMR-HR32                | 3DPW      |73.9   | 44.9     | [Download]()|

### Qualitative results

Results on **3DPW** dataset:

<p align="center"><img src='documents/fig3.jpg'></p>

Results on **COCO** dataset:

<p align="center"><img src='documents/fig4.jpg'></p>

## License

This research code is released under the MIT license. Please see [LICENSE](LICENSE) for more information.

SMPL and MANO models are subject to **Software Copyright License for non-commercial scientific research purposes**. Please see [SMPL-Model License](https://smpl.is.tue.mpg.de/modellicense.html) and [MANO License](https://mano.is.tue.mpg.de/license.html) for more information.

We use submodules from third party ([hassony2/manopth](https://github.com/hassony2/manopth)). Please see [NOTICE](documents/NOTICE.md) for more information.


## Acknowledgments
This work was supported by Institute of Information \& communications Technology Planning \& Evaluation(IITP) grant funded by the Korea government(MSIT) (2021-0-02084, eXtended Reality and Volumetric media generation and transmission technology for immersive experience sharing in noncontact environment with a Korea-EU international cooperative research).

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[microsoft/MeshTransformer](https://github.com/microsoft/MeshTransformer)  </br>
[microsoft/MeshGraphormer](https://github.com/microsoft/MeshGraphormer)  </br>
[postech-ami/FastMETRO](https://github.com/postech-ami/FastMETRO)  </br>
[Arthur151/ROMP](https://github.com/Arthur151/ROMP)  </br>



## Citation
```bibtex
@InProceedings{PointHMR,
author = {Kim, Jeonghwan and Gwon, Mi-Gyeong and Park, Hyunwoo and Kwon, Hyukmin and Um, Gi-Mun and Kim, Wonjun},
title = {{Sampling is Matter}: Point-guided 3D Human Mesh Reconstruction},
booktitle = {CVPR},
month = {June},
year = {2023}
}
```
<!--
## License
 -->
