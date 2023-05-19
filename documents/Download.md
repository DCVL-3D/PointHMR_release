## Download

1. Create directories for pretrained models and datasets.

   ```bash
   export REPO_DIR=$PWD
   mkdir -p $REPO_DIR/models  # pre-trained models
   mkdir -p $REPO_DIR/datasets  # datasets
   ```

2. Download pretrained backbone and model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1FnDfG9CS6KrTSUM4L-Vb4Q5V9-92-SSx)

   ***The weights of pretrained backbones are provided from the previous project [ROMP](https://github.com/Arthur151/ROMP)***

   

   The resulting data structure should follow the hierarchy as below.

   ```
   ${REPO_DIR}  
   |-- models  
   |   |-- PointHMR_h36m.bin
   |   |-- PointHMR_3dpw.bin
   |   |-- pretrain_hrnet.pkl
   |   |-- pretrain_resnet.pkl
   |-- src 
   |-- datasets 
   |-- predictions 
   |-- README.md 
   |-- ... 
   |-- ... 
   ```

3. Download SMPL and MANO models from their official websites

   To run our code smoothly, please visit the following websites to download SMPL and MANO models.

   - Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [SMPLify](http://smplify.is.tue.mpg.de/), and place it at `${REPO_DIR}/src/modeling/data`.
   - Download `MANO_RIGHT.pkl` from [MANO](https://mano.is.tue.mpg.de/), and place it at `${REPO_DIR}/src/modeling/data`.

   Please put the downloaded files under the `${REPO_DIR}/src/modeling/data` directory. The data structure should follow the hierarchy below.

   ```
   ${REPO_DIR}  
   |-- src  
   |   |-- modeling
   |   |   |-- data
   |   |   |   |-- basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
   |   |   |   |-- MANO_RIGHT.pkl
   |-- models
   |-- datasets
   |-- predictions
   |-- README.md 
   |-- ... 
   |-- ... 
   ```

   Please check [/src/modeling/data/README.md](https://github.com/microsoft/MeshGraphormer/blob/main/src/modeling/data/README.md) for further details.

   

4. Download datasets and pseudo labels for training.

   ***We use the same data from the previous project [METRO](https://github.com/microsoft/MeshTransformer)***

   

   We recommend to download large files with **AzCopy** for faster speed. AzCopy executable tools can be downloaded [here](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy). Decompress the azcopy tar file and put the executable in any path.

   To download the annotation files, please use the following command.

   ```
   cd $REPO_DIR
   path/to/azcopy copy 'https://datarelease.blob.core.windows.net/metro/datasets/filename.tar' /path/to/your/folder/filename.tar
   tar xvf filename.tar  
   ```

   

   `filename.tar` could be `Tax-H36m-coco40k-Muco-UP-Mpii.tar`, `human3.6m.tar`, `coco_smpl.tar`, `muco.tar`, `up3d.tar`, `mpii.tar`, `3dpw.tar`, `freihand.tar`. Total file size is about 200 GB.

   The datasets and pseudo ground truth labels are provided by [Pose2Mesh](https://github.com/hongsukchoi/Pose2Mesh_RELEASE). We only reorganize the data format to better fit our training pipeline. We suggest to download the orignal image files from the offical dataset websites.

   The `datasets` directory structure should follow the below hierarchy.

   ```
   ${ROOT}  
   |-- models 
   |-- src
   |-- datasets  
   |   |-- Tax-H36m-coco40k-Muco-UP-Mpii  
   |   |   |-- train.yaml 
   |   |   |-- train.linelist.tsv  
   |   |   |-- train.linelist.lineidx
   |   |-- human3.6m  
   |   |   |-- train.img.tsv 
   |   |   |-- train.hw.tsv 
   |   |   |-- train.linelist.tsv    
   |   |   |-- smpl/train.label.smpl.p1.tsv
   |   |   |-- smpl/train.linelist.smpl.p1.tsv
   |   |   |-- valid.protocol2.yaml
   |   |   |-- valid_protocol2/valid.img.tsv 
   |   |   |-- valid_protocol2/valid.hw.tsv  
   |   |   |-- valid_protocol2/valid.label.tsv
   |   |   |-- valid_protocol2/valid.linelist.tsv
   |   |-- coco_smpl  
   |   |   |-- train.img.tsv  
   |   |   |-- train.hw.tsv   
   |   |   |-- smpl/train.label.tsv
   |   |   |-- smpl/train.linelist.tsv
   |   |-- muco  
   |   |   |-- train.img.tsv  
   |   |   |-- train.hw.tsv   
   |   |   |-- train.label.tsv
   |   |   |-- train.linelist.tsv
   |   |-- up3d  
   |   |   |-- trainval.img.tsv  
   |   |   |-- trainval.hw.tsv   
   |   |   |-- trainval.label.tsv
   |   |   |-- trainval.linelist.tsv
   |   |-- mpii  
   |   |   |-- train.img.tsv  
   |   |   |-- train.hw.tsv   
   |   |   |-- train.label.tsv
   |   |   |-- train.linelist.tsv
   |   |-- 3dpw 
   |   |   |-- train.img.tsv  
   |   |   |-- train.hw.tsv   
   |   |   |-- train.label.tsv
   |   |   |-- train.linelist.tsv
   |   |   |-- test_has_gender.yaml
   |   |   |-- has_gender/test.img.tsv 
   |   |   |-- has_gender/test.hw.tsv  
   |   |   |-- has_gender/test.label.tsv
   |   |   |-- has_gender/test.linelist.tsv
   |   |-- freihand
   |   |   |-- train.yaml
   |   |   |-- train.img.tsv  
   |   |   |-- train.hw.tsv   
   |   |   |-- train.label.tsv
   |   |   |-- train.linelist.tsv
   |   |   |-- test.yaml
   |   |   |-- test.img.tsv  
   |   |   |-- test.hw.tsv   
   |   |   |-- test.label.tsv
   |   |   |-- test.linelist.tsv
   |-- README.md 
   |-- ... 
   |-- ... 
   ```

   
