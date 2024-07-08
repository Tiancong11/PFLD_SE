#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from torch.nn import Module, AvgPool2d, Linear
from models.base_module import Conv_Block, InvertedResidual

# 宽度因子，宽度多少倍。用来改变模型通道数（模型复杂程度）扩展或缩减网络的每一层的通道数。例如，如果你有一个基础模型，
# 并且width_factor是1，那么模型的每一层通道数保持不变；如果width_factor是2，那么每一层的通道数会翻倍，这样可以增加模型的容量和性能。设置为1表示使用默认的通道数。

#  group=int(64 * width_factor)表示每个输出通道被划分为多少个分组（也叫作通道组或组内卷积）。

class PFLD(Module):
    def __init__(self, width_factor=1, input_size=112, landmark_number=98):
        super(PFLD, self).__init__()

        self.conv1 = Conv_Block(3, int(64 * width_factor), 3, 2, 1)
        self.conv2 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1, group=int(64 * width_factor))
        # 倒数第二个False是用来控制在神经网络中是否使用残差连接（Residual Connection）的参数。
        # 残差连接是一种常用的技术，特别是在深度神经网络（如深层卷积神经网络）中，用于解决梯度消失和梯度爆炸问题，并帮助网络更好地学习和训练。
        self.conv3_1 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 2, False, 2)
        self.conv3_2 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 1, True, 2)
        self.conv3_3 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 1, True, 2)
        self.conv3_4 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 1, True, 2)
        self.conv3_5 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 1, True, 2)

        self.conv4 = InvertedResidual(int(64 * width_factor), int(128 * width_factor), 2, False, 2)

        self.conv5_1 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, False, 4)
        self.conv5_2 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)
        self.conv5_3 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)
        self.conv5_4 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)
        self.conv5_5 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)
        self.conv5_6 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)

        self.conv6 = InvertedResidual(int(128 * width_factor), int(16 * width_factor), 1, False, 2)
        self.conv7 = Conv_Block(int(16 * width_factor), int(32 * width_factor), 3, 2, 1)
        self.conv8 = Conv_Block(int(32 * width_factor), int(128 * width_factor), input_size // 16, 1, 0, has_bn=False)

        self.avg_pool1 = AvgPool2d(input_size // 8)
        self.avg_pool2 = AvgPool2d(input_size // 16)
        self.fc = Linear(int(176 * width_factor), landmark_number * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv3_5(x)

        x = self.conv4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x = self.conv5_5(x)
        x = self.conv5_6(x)

        x = self.conv6(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.conv8(x)
        x3 = x3.view(x1.size(0), -1)
        # cat按照第一个维度拼接

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return landmarks


class PFLD_Ultralight_AuxiliaryNet(Module):
    def __init__(self, width_factor=1):
        # super(PFLD_Ghost_AuxiliaryNet, self).__init__()
        super(PFLD_Ultralight_AuxiliaryNet, self).__init__()

        self.conv1 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 1, 1, 0)
        self.conv2 = Conv_Block(int(80 * width_factor), int(64 * width_factor), 1, 1, 0)
        self.conv3 = Conv_Block(int(96 * width_factor), int(64 * width_factor), 1, 1, 0)
        self.conv4 = Conv_Block(int(144 * width_factor), int(64 * width_factor), 1, 1, 0)

        self.merge1 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1)
        self.merge2 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1)
        self.merge3 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1)

        self.conv_out = Conv_Block(int(64 * width_factor), 1, 1, 1, 0)

    def forward(self, out1, out2, out3, out4):
        output1 = self.conv1(out1)
        output2 = self.conv2(out2)
        output3 = self.conv3(out3)
        output4 = self.conv4(out4)

        up4 = F.interpolate(output4, size=[output3.size(2), output3.size(3)], mode="nearest")
        output3 = output3 + up4
        output3 = self.merge3(output3)

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        output1 = self.conv_out(output1)

        return output1
