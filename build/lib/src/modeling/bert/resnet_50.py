from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import torchvision.transforms.functional as F
import sys, os
import logging
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.modeling.bert.basic_modules import BasicBlock,Bottleneck,HighResolutionModule
BN_MOMENTUM = 0.1

class ResNet_50(nn.Module):
    def __init__(self, args, lap_num=1, **kwargs):
        self.inplanes = 64
        self.device = args.device
        super(ResNet_50, self).__init__()
        self.make_resnet()
        self.backbone_channels = 64
        self.init_weights()
        self.load_pretrain_params()



    def load_pretrain_params(self):
        if os.path.exists('./models/pretrain_resnet.pkl'):
            success_layer = copy_state_dict(self.state_dict(), torch.load('./models/pretrain_resnet.pkl', map_location=self.device), prefix = '', fix_loaded=True)
            print("load success")
    def image_preprocess(self, x):
        if args().pretrain == 'imagenet' or args().pretrain == 'spin':
            x = BHWC_to_BCHW(x)/255.
            #x = F.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],inplace=True).contiguous() # for pytorch version>1.8.0
            x = torch.stack(list(map(lambda x:F.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],inplace=False),x)))
        else:
            x = ((BHWC_to_BCHW(x)/ 255.) * 2.0 - 1.0).contiguous()

        return x

    def make_resnet(self):
        block, layers = Bottleneck, [3, 4, 6, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_resnet_layer(block, 64, layers[0])
        self.layer2 = self._make_resnet_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_resnet_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_resnet_layer(block, 512, layers[3], stride=2)

        self.deconv_layers = self._make_deconv_layer(3, (256, 128, 64), (4, 4, 4))
        # if self.lap_num == 1:
        #     self.deconv_layers = self._make_deconv_layer(2,(1024,1024),(4,4))
        # if self.lap_num == 2:
        #     self.deconv_layers = self._make_deconv_layer(2,(1024,1024),(3,4))
        # if self.lap_num == 3:
        #     self.deconv_layers = self._make_deconv_layer(2,(1024,1024),(2,3))

    def forward(self,x):
        # x = self.image_preprocess(x)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_ = self.layer4(x)
        # out = []
        # if self.lap_num > 1:
        x = self.deconv_layers(x_)
        return x, x_

    def _make_resnet_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),)#,affine=False),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            if i == 0:
                self.inplanes = 2048
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))  # ,affine=False))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

def copy_state_dict(cur_state_dict, pre_state_dict, prefix = 'module.', drop_prefix='', fix_loaded=False):
    success_layers, failed_layers = [], []
    def _get_params(key):
        key = key.replace(drop_prefix,'')
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                failed_layers.append(k)
                continue
            cur_state_dict[k].copy_(v)
            if prefix in k and prefix!='':
                k=k.split(prefix)[1]
            success_layers.append(k)
        except:
            logging.info('copy param {} failed, mismatched'.format(k))
            failed_layers.append(k)
            continue
    logging.info('missing parameters of layers:{}'.format(failed_layers))

    if fix_loaded and len(failed_layers)>0:
        print('fixing the layers that were loaded successfully, while train the layers that failed,')
        for k in cur_state_dict.keys():
            try:
                if k in success_layers:
                    cur_state_dict[k].requires_grad=False
            except:
                print('fixing the layer {} failed'.format(k))

    del pre_state_dict
    return success_layers
if __name__ == '__main__':
    args().pretrain = 'spin'
    model = ResNet_50().cuda()
    a=model(torch.rand(2,512,512,3).cuda())
    for i in a:
        print(i.shape)