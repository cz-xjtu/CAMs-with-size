import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torchvision.models import ResNet
import torch
import torch.nn.functional as F

__all__ = ['Net', 'resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'resnet_152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Net(nn.Module):

    def __init__(self, block, layer_num, num_classes=2):
        self.in_planes = 64
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layer_num[0])
        self.layer2 = self._make_layer(block, 128, layer_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_num[3], stride=2)
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.globalMaxPool = nn.AdaptiveMaxPool2d(1)
        self.fc_cz = nn.Linear(512 * block.expansion, num_classes)
        self.class_num = num_classes
        self.sigmoid = nn.Sigmoid()

        """initializing the model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, block_num):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.globalAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_cz(x)

        return x

    def forward_cam(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.sigmoid(x)
        # conv5_3_3 = self.layer4[2].conv3(x)
        conv5_3_3 = x
        bz, nc, h, w = conv5_3_3.shape

        x = self.globalAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_cz(x)
        cams = [torch.mm(self.fc_cz.weight[0].unsqueeze(0), conv5_3_3[batch_id].view(nc, h * w)).view(-1)
                for batch_id in range(x.shape[0])]
        cams = torch.cat(cams, dim=0).view(bz, self.class_num, h, w)
        '''cams_min = [torch.min(cams[batch_id][0]).view(1, 1, 1, 1) for batch_id in range(x.shape[0])]
        cams_min = torch.cat(cams_min, 0)
        cams_max = self.globalMaxPool(cams)
        cams = (cams - cams_min) / (cams_max - cams_min)'''
        #cams = torch.sigmoid(cams)
        cams = F.interpolate(cams, scale_factor=32, mode='bilinear', align_corners=True)
        # cams = torch.nn.Upsample(scale_factor=32, mode='bilinear')(cams)
        return x, conv5_3_3, cams


def resnet_18(pretrained=False, **kwargs):

    model = Net(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet_34(pretrained=False, **kwargs):
    model = Net(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet_50(pretrained=False, **kwargs):
    model = Net(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    if pretrained:
        model_dict = model.state_dict()
        checkpoint = torch.load('pretrained/resnet50-19c8e357.pth')
        # 1. filter out unnecessary keys
        base_dict = {'.'.join(k.split('.')[0:]): v for k, v in list(checkpoint.items()) if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(base_dict)
        # 2. overwrite entries in the existing state dict
        model.load_state_dict(model_dict)
        '''for k, v in model.named_parameters():
            if k != 'fc_cz.weight' or k != 'fc_cz.bias':
                v.requires_grad = False'''
    return model


def resnet_101(pretrained=False, **kwargs):
    model = Net(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet_152(pretrained=True, **kwargs):
    model = Net(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        checkpoint = torch.load('pretrained/resnet152-b121ed2d.pth')
        # 1. filter out unnecessary keys
        base_dict = {'.'.join(k.split('.')[0:]): v for k, v in list(checkpoint.items()) if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(base_dict)
        # 2. overwrite entries in the existing state dict
        model.load_state_dict(model_dict)
        '''
        for k, v in model.named_parameters():
            #print(k)
            if k != 'fc_cz.weight' and k != 'fc_cz.bias':
                v.requires_grad = False
        '''

    return model


