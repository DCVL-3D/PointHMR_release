## Experiments

Our PointHMR can be run by following command

### Pretraining scheme

We conduct large-scale training on multiple 2D and 3D datasets, including Human3.6M, COCO, MUCO, UP3D, MPII. During training, it will evaluate the performance per epoch, and save the best checkpoints.

```
sh script/pretrain.sh
```

- nproc_per_node: Number of GPU.

- train_yaml: Path of training datasets (Human3.6M, COCO, MUCO, UP3D, MPII).

- val_yaml: Path of valiadation dataset (Human3.6M).

  

### Finetuning scheme

We follow prior works that also use 3DPW training data.  During training, it will evaluate the performance per epoch, and save the best checkpoints. We fine-tune on 3DPW training set with the checkpoint from the pretraining scheme by the following command.

```
sh script/finetuning.sh
```

- nproc_per_node: Number of GPU.

- train_yaml: Path of training dataset (3DPW).

- val_yaml: Path of validation dataset (3DPW).

- resume_checkpoint: Path to the specific checkpoint (checkpoint from pretraining) for resume training.

  

### Evaluation Human3.6M

We evaluate our PointHMR on the Human3.6M dataset by the following command.

```
sh script/evaluation_h36m.sh
```

- nproc_per_node: Number of GPU.

- val_yaml: Path of validation dataset (Human3.6M).

- resume_checkpoint: Path to the specific checkpoint for evaluation.

  

### Evaluation 3DPW

We evaluate our PointHMR on the 3DPW dataset by the following command.

```
sh script/evalutation_3dpw.sh
```

- nproc_per_node: Number of GPU.
- val_yaml: Path of validation dataset (3DPW).
- resume_checkpoint: Path to the specific checkpoint for evaluation.
