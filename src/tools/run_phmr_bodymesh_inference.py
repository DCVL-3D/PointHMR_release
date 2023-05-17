"""
----------------------------------------------------------------------------------------------
PointHMR Official Code
Copyright (c) Deep Computer Vision Lab. (DCVL.) All Rights Reserved
Licensed under the MIT license.
----------------------------------------------------------------------------------------------
Modified from MeshGraphormer (https://github.com/microsoft/MeshGraphormer)
Copyright (c) Microsoft Corporation. All Rights Reserved [see https://github.com/microsoft/MeshGraphormer/blob/main/LICENSE for details]
----------------------------------------------------------------------------------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as op
import pickle
import torch
import gc
import numpy as np
import cv2
from src.modeling.model.network import PointHMR
from src.modeling._smpl import SMPL, Mesh

from src.utils.logger import setup_logger
from src.utils.comm import get_rank
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.renderer import Renderer, visualize_reconstruction_no_text

from PIL import Image
from torchvision import transforms
smpl_face = pickle.load(open('src/modeling/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl','rb'), encoding='latin1')['f'].astype(np.int)

def save_obj(verts, faces=smpl_face, obj_mesh_name='mesh.obj'):
    #print('Saving:',obj_mesh_name)
    with open(obj_mesh_name, 'w') as fp:
        for v in verts:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

        for f in faces: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )


transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def run_inference(args, image_list, Graphormer_model, smpl, renderer, mesh_sampler):
    # switch to evaluate mode
    Graphormer_model.eval()
    smpl.eval()

    create_directory_if_not_exists(args.image_output_dir)

    with torch.no_grad():
        for image_file in image_list:
            if 'pred' not in image_file:
                att_all = []
                img = Image.open(image_file)
                img_size = img.size  # 이미지의 크기 측정
                # 직사각형의 이미지가 256x512 이라면, img_size = (256,512)가 된다.
                x = img_size[0]  # 넓이값
                y = img_size[1]  # 높이값

                # if x != y:
                #     size = max(x, y)
                #     resized_img = Image.new(mode='RGB', size=(size, size), color=(0))
                #     offset = (round((abs(x - size)) / 2), round((abs(y - size)) / 2))
                #     resized_img.paste(img, offset)
                #     img = resized_img

                img_tensor = transform(img)
                img_visual = transform_visualize(img)

                batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
                batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()
                # forward-pass
                outputs = Graphormer_model(batch_imgs)

                pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, heat_map = outputs

                # obtain 3d joints from full mesh
                visual_imgs_output = visualize_mesh(renderer, batch_visual_imgs[0],
                                                    pred_vertices[0].detach(),
                                                    pred_camera.squeeze(0).detach())

                visual_imgs = visual_imgs_output.transpose(1, 2, 0)
                visual_imgs = np.asarray(visual_imgs)

                filename = os.path.basename(image_file)[:-4]

                temp_fname = os.path.join(args.image_output_dir, filename + '.jpg')
                cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))
    return 

def visualize_mesh( renderer, images,
                    pred_vertices_full,
                    pred_camera):
    img = images.cpu().numpy().transpose(1,2,0)
    # Get predict vertices for the particular example
    vertices_full = pred_vertices_full.cpu().numpy() 
    cam = pred_camera.cpu().numpy()
    # Visualize only mesh reconstruction 
    rend_img = visualize_reconstruction_no_text(img, 224, vertices_full, cam, renderer, color='light_blue')
    rend_img = rend_img.transpose(2,0,1)
    return rend_img

def visualize_mesh_and_attention( renderer, images,
                    pred_vertices_full,
                    pred_vertices, 
                    pred_2d_vertices,
                    pred_2d_joints,
                    pred_camera,
                    attention):
    img = images.cpu().numpy().transpose(1,2,0)
    # Get predict vertices for the particular example
    vertices_full = pred_vertices_full.cpu().numpy() 
    vertices = pred_vertices.cpu().numpy()
    vertices_2d = pred_2d_vertices.cpu().numpy()
    joints_2d = pred_2d_joints.cpu().numpy()
    cam = pred_camera.cpu().numpy()
    att = attention.cpu().numpy()
    # Visualize reconstruction and attention
    rend_img = visualize_reconstruction_and_att_local(img, 224, vertices_full, vertices, vertices_2d, cam, renderer, joints_2d, att, color='light_blue')
    rend_img = rend_img.transpose(2,0,1)
    return rend_img


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--num_workers", default=4, type=int, 
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.")
    parser.add_argument("--image_file_or_path", default='./samples/human-body', type=str, 
                        help="test data") 
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='src/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--image_output_dir", default='demo/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=30, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=30, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float,
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=200, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vertices_loss_weight", default=100.0, type=float)
    parser.add_argument("--joints_loss_weight", default=1000.0, type=float)
    parser.add_argument("--heatmap_loss_weight", default=100.0, type=float)

    parser.add_argument("--vloss_w_full", default=0.33, type=float)
    parser.add_argument("--vloss_w_sub", default=0.33, type=float)
    parser.add_argument("--vloss_w_sub2", default=0.33, type=float)
    parser.add_argument("--drop_out", default=0.1, type=float,
                        help="Drop out ratio in BERT.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument("--transformer_nhead", default=4, type=int, required=False,
                        help="Update model config if given. Note that the division of "
                             "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--model_dim", default=512, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--feedforward_dim_1", default=1024, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--feedforward_dim_2", default=512, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--position_dim", default=128, type=int,
                        help="position dim.")
    parser.add_argument("--activation", default="relu", type=str,
                        help="The Image Feature Dimension.")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="The Image Feature Dimension.")
    parser.add_argument("--mesh_type", default='body', type=str, help="body or hand")
    parser.add_argument("--interm_size_scale", default=2, type=int)
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true', )
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88,
                        help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training.")

    args = parser.parse_args()
    return args


def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))

    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    mkdir(args.output_dir)
    logger = setup_logger("Graphormer", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()

    # Renderer for visualization
    renderer = Renderer(faces=smpl.faces.cpu().numpy())

    if args.run_eval_only == True and args.resume_checkpoint != None and args.resume_checkpoint != 'None' and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _model = torch.load(args.resume_checkpoint)
    else:
        _model = PointHMR(args, mesh_sampler)

        if args.resume_checkpoint != None and args.resume_checkpoint != 'None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            # workaround approach to load sparse tensor in graph conv.
            states = torch.load(args.resume_checkpoint, map_location=args.device)

            for k, v in states.items():
                states[k] = v.cpu()
            _model.load_state_dict(states, strict=False)

            del states
            gc.collect()
            torch.cuda.empty_cache()

    _model.to(args.device)
    logger.info("Run inference")

    image_list = []
    # args.image_file_or_path = sorted(args.image_file_or_path)
    if not args.image_file_or_path:
        raise ValueError("image_file_or_path not specified")
    if op.isfile(args.image_file_or_path):
        image_list = [args.image_file_or_path]
    elif op.isdir(args.image_file_or_path):
        # should be a path with images only
        for filename in os.listdir(args.image_file_or_path):
            if filename.endswith(".png") or filename.endswith(".jpg") and 'pred' not in filename:
                image_list.append(args.image_file_or_path + '/' + filename)
    else:
        raise ValueError("Cannot find images at {}".format(args.image_file_or_path))
    image_list = sorted(image_list)

    run_inference(args, image_list, _model, smpl, renderer, mesh_sampler)


if __name__ == "__main__":
    args = parse_args()
    main(args)
