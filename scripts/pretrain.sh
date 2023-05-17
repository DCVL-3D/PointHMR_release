python -m torch.distributed.launch --nproc_per_node=2 \
       src/tools/run_phmr_bodymesh.py \
       --train_yaml ../dataset/Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml \
       --val_yaml ../dataset/human3.6m/valid.protocol2.yaml \
       --num_workers 4 \
       --per_gpu_train_batch_size 12 \
       --per_gpu_eval_batch_size 12 \
       --model_dim 512 \
       --position_dim 128 \
       --dropout 0.1 \
       --learning_rate 1e-5 \
       --num_train_epochs 60 \

