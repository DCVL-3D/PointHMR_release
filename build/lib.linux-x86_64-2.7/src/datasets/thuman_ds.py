"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
import os
import cv2
import math
import json
from PIL import Image
import os.path as op
import numpy as np
import code

from src.modeling._smpl import SMPL
from src.utils.image_ops import img_from_base64, crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import torch
import torchvision.transforms as transforms

import glob
class THumanDataset(object):
    def __init__(self, img_file, label_file=None, hw_file=None,
                 linelist_file=None, is_train=True, cv2_output=False, scale_factor=1):

        self.img_file = img_file
        self.label_file = label_file
        # self.hw_file = hw_file
        # self.linelist_file = linelist_file
        # self.img_tsv = self.get_tsv_file(img_file)
        # self.label_tsv = None if label_file is None else self.get_tsv_file(label_file)
        # self.hw_tsv = None if hw_file is None else self.get_tsv_file(hw_file)
        self.is_train = is_train

        self.file_path = sorted(glob.glob("../dataset/THuman/data/*"))


        self.smpl_path = '../dataset/THuman/THuman2.0_smplx/'
        if self.is_train:
            self.file_path = self.file_path[:-50]
        else:
            self.file_path = self.file_path[:]

        self.cv2_output = cv2_output
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.scale_factor = 0.25 # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.noise_factor = 0.4
        self.rot_factor = 30 # Random rotation in the range [-rot_factor, rot_factor]
        self.img_res = 224

        # self.image_keys = self.prepare_image_keys()

        self.joints_definition = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder',
        'L_Elbow','L_Wrist','Neck','Top_of_Head','Pelvis','Thorax','Spine','Jaw','Head','Nose','L_Eye','R_Eye','L_Ear','R_Ear')
        self.pelvis_index = self.joints_definition.index('Pelvis')

        self.smpl_regress = SMPL()


    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.get_key(i) : i for i in range(tsv.num_rows())}


    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
	    
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
	    
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
	    
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.scale_factor,
                    max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
	
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, flip, pn):
        """Process rgb image and do augmentation."""
        # flip the image 
        # if flip:
        #     rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    # def j2d_processing(self, kp, center, scale, r, f):
    #     """Process gt 2D keypoints and apply all augmentation transforms."""
    #     nparts = kp.shape[0]
    #     for i in range(nparts):
    #         kp[i,0:2] = transform(kp[i,0:2]+1, center, scale,
    #                               [self.img_res, self.img_res], rot=r)
    #     # convert to normalized coordinates
    #     kp[:,:-1] = 2.*kp[:,:-1]/self.img_res - 1.
    #     # flip the x coordinates
    #     if f:
    #          kp = flip_kp(kp)
    #     kp = kp.astype('float32')
    #     return kp

    def j3d_processing(self, S, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        # if not r == 0:
        #     rot_rad = -r * np.pi / 180
        #     sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        #     rot_mat[0,:2] = [cs, -sn]
        #     rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose = pose.astype('float32')
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def get_image(self, idx):
        rot_n = 0
        img_path = os.path.join(self.file_path[idx], f"image/{rot_n:04d}.jpg")

        cv2_im = cv2.imread(img_path)
        if self.cv2_output:
            return cv2_im.astype(np.float32, copy=True)
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

        return cv2_im, rot_n

    def get_target_from_annotations(self, annotations, img_size, idx):
        # This function will be overwritten by each dataset to 
        # decode the labels to specific formats for each task. 
        return annotations


    def get_img_info(self, idx):
        if self.hw_tsv is not None:
            line_no = self.get_line_no(idx)
            row = self.hw_tsv[line_no]
            try:
                # json string format with "height" and "width" being the keys
                return json.loads(row[1])[0]
            except ValueError:
                # list of strings representing height and width in order
                hw_str = row[1].split(' ')
                hw_dict = {"height": int(hw_str[0]), "width": int(hw_str[1])}
                return hw_dict

    def get_img_key(self, idx):
        base_num = os.path.basename(self.file_path[idx])
        key_path = os.path.join(self.smpl_path, f'{base_num}/smplx_param.pkl')
        #THuman2.0_smplx/0525/smplx_param.pkl"
        db = np.load(key_path, allow_pickle=True)
        dict_ = {}
        for key in db:
            dict_[key] = torch.tensor(db[key])

        return dict_

    def __len__(self):

        return len(self.file_path)

    def __getitem__(self, idx):

        img, rot_n = self.get_image(idx)
        annot = self.get_img_key(idx)
        # go = annot["global_orient"]
        go = torch.zeros(1,3)
        # print(go.shape)
        # go[0][0] += torch.pi
        # torch.cat([go, dict_["body_pose"][0], torch.zeros(6)],0)
        pose = torch.cat([go[0], annot["body_pose"][0], torch.zeros(6)], 0).reshape(1, -1).float()
        betas = annot["betas"].reshape(1, -1).float()

        verts = self.smpl_regress(pose, betas)

        # verts[0, :, 1] *= -1

        joints_3d = self.smpl_regress.get_h36m_joints(verts)

        has_2d_joints = [False]
        has_3d_joints = [True]
        joints_2d = np.asarray([None,None,None])

        if joints_2d.ndim==3:
            joints_2d = joints_2d[0]
        if joints_3d.ndim==3:
            joints_3d = joints_3d[0]

        # Get SMPL parameters, if available
        has_smpl = np.asarray([True])
        gender = 'none'

        # Get augmentation parameters
        flip,pn,rot,sc = self.augm_params()

        # Process image
        img = self.rgb_processing(img, flip, pn)
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        transfromed_img = self.normalize_img(img)

        # normalize 3d pose by aligning the pelvis as the root (at origin)
        root_pelvis = joints_3d[self.pelvis_index,:-1]
        joints_3d[:,:-1] = joints_3d[:,:-1] - root_pelvis[None,:]
        # 3d pose augmentation (random flip + rotation, consistent to image and SMPL)
        # joints_3d_transformed = self.j3d_processing(joints_3d.numpy().copy(), flip)
        # 2d pose augmentation
        # joints_2d_transformed = self.j2d_processing(joints_2d.copy(), center, sc*scale, rot, flip)

        ###################################
        # Masking percantage
        # We observe that 30% works better for human body mesh. Further details are reported in the paper.
        mvm_percent = 0.3
        ###################################
        
        mjm_mask = np.ones((14,1))
        if self.is_train:
            num_joints = 14
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_joints) # at most x% of the joints could be masked
            indices = np.random.choice(np.arange(num_joints),replace=False,size=masked_num)
            mjm_mask[indices,:] = 0.0
        mjm_mask = torch.from_numpy(mjm_mask).float()

        mvm_mask = np.ones((431,1))
        if self.is_train:
            num_vertices = 431
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_vertices) # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_vertices),replace=False,size=masked_num)
            mvm_mask[indices,:] = 0.0
        mvm_mask = torch.from_numpy(mvm_mask).float()

        meta_data = {}
        meta_data['ori_img'] = img
        meta_data['pose'] = pose[0]
        meta_data['betas'] = betas[0]
        meta_data['joints_3d'] = torch.from_numpy(joints_3d.numpy()).float()
        meta_data['has_3d_joints'] = torch.tensor(has_3d_joints)
        meta_data['has_smpl'] = torch.tensor(has_smpl)

        meta_data['mjm_mask'] = mjm_mask
        meta_data['mvm_mask'] = mvm_mask

        # Get 2D keypoints and apply augmentation transforms
        meta_data['has_2d_joints'] = torch.tensor(has_2d_joints)
        meta_data['joints_2d'] = torch.from_numpy(joints_3d.numpy()).float()[:,:2]
        # meta_data['scale'] = float(sc * scale)
        # meta_data['center'] = np.asarray(center).astype(np.float32)
        meta_data['gender'] = gender
        return torch.tensor([0]), transfromed_img, meta_data

