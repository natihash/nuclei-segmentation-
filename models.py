import torch
import torch.nn as nn
# import torchvision.transforms.functional as TF
import cv2
import glob 
import os
# import xml.etree.ElementTree as ET
import numpy as np
# import matplotlib.pyplot as plt
import random
# import torch.nn.functional as F
from torch.utils.data import Dataset

# class VGGBlock(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels):
#         super(VGGBlock, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, middle_channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(middle_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(middle_channels, out_channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.conv(x)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False)
        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.identity(x)
        y = self.BN(y)
        y = self.relu(y)
        return torch.add(self.conv(x), y)


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels, deep_supervision):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.deep_supervision = deep_supervision
        # self.guts = num_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        
        self.conv0_0_2 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv0_1_2 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_3_2 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_4_2 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes[0], kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes[0], kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes[0], kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes[0], kernel_size=1)
            
            self.final1_2 = nn.Conv2d(nb_filter[0], num_classes[1], kernel_size=1)
            self.final2_2 = nn.Conv2d(nb_filter[0], num_classes[1], kernel_size=1)
            self.final3_2 = nn.Conv2d(nb_filter[0], num_classes[1], kernel_size=1)
            self.final4_2 = nn.Conv2d(nb_filter[0], num_classes[1], kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes[0], kernel_size=1)
            self.final_2 = nn.Conv2d(nb_filter[0], num_classes[1], kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x0_0_2 = self.conv0_0_2(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_1_2 = self.conv0_1_2(torch.cat([x0_0_2, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x0_2_2 = self.conv0_2_2(torch.cat([x0_0_2, x0_1_2, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x0_3_2 = self.conv0_3_2(torch.cat([x0_0_2, x0_1_2, x0_2_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        x0_4_2 = self.conv0_4_2(torch.cat([x0_0_2, x0_1_2, x0_2_2, x0_3_2, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)

            output1_2 = self.final1_2(x0_1_2)
            output2_2 = self.final2_2(x0_2_2)
            output3_2 = self.final3_2(x0_3_2)
            output4_2 = self.final4_2(x0_4_2)
            # return [output1, output2, output3, output4], [output1_2, output2_2, output3_2, output4_2]
            return (output1+output2+output3+output4)/4, (output1_2+output2_2+output3_2+output4_2)/4
        else:
            output = self.final(x0_4)
            output_2 = self.final_2(x0_4_2)
            return output, output_2
