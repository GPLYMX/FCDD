# -*- coding: utf-8 -*-
# @Author : guopeng
# @Version : 2.1
# @Time : 2022/11/15 14:42:25
import os
# from unittest import result
import torch
import torch.nn as nn
from torchvision import transforms
# from algo_deploy.sz_color.models.fcdd_cnn_224 import *
from models.fcdd_cnn_224 import *
from PIL import Image
import time
import inspect
import sys
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm


def proprecess_fcdd(data):
    test_transform = [
        transforms.Resize((248)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3798, 0.2742, 0.2732],
                             std=[0.3175, 0.2436, 0.2463])
    ]
    return transforms.Compose(test_transform)(data)


class ModelClass2_fcdd(nn.Module):
    def all_nets(self) -> Dict[str, Tuple[torch.nn.Module, type]]:
        """ returns a mapping form model names to a tuple of (model instance, corresponding AE class) """
        members = inspect.getmembers(sys.modules[__name__])
        # print("members:",members)
        clsses = {
            name: ((obj, obj.encoder_cls) if hasattr(obj, 'encoder_cls') else (obj, None))
            for name, obj in members if inspect.isclass(obj)
        }
        return clsses

    def load_nets(self, **kwargs) -> torch.nn.Module:
        name = 'FCDD_CNN224_VGG_F'
        in_shape = (3, 224, 224)
        bias = True
        NET, _ = self.all_nets()[name]
        net = NET(in_shape, bias=bias, **kwargs)

        return net


def infer_fcdd(img, model):
    """
    说明1:img格式为RGB(PIL),单张图片
    """
    index2label = {0: "正常", 1: "异常"}
    outputs = model(img)
    loss = outputs ** 2
    loss = (loss + 1).sqrt() - 1
    red_ascores = loss.reshape(loss.size(0), -1).mean(1)
    # print("red_ascores:",red_ascores)
    if red_ascores < 0.1:
        pred = int(0)
    else:
        pred = int(1)
    label = index2label[pred]

    return pred, label


class DeploySZColor(object):
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cpu")  # 限定使用cpu

        # 当前文件所在目录
        path_cur = os.path.dirname(__file__)

        # 异常检测 构建模型
        self.model_fcdd = ModelClass2_fcdd().load_nets()
        snapshot = torch.load(os.path.join(path_cur, "snapshot.pt"), map_location=self.device)
        net_state = snapshot.pop('net', None)
        self.model_fcdd.load_state_dict(net_state)
        self.model_fcdd.eval()
        self.model_fcdd.to(device=self.device)

    def run(self, img: np.ndarray):
        # 单张图片的格式统一为np.ndarray：[h,w,3]
        img = Image.fromarray(img).convert("RGB")

        # 图片预处理
        img1 = proprecess_fcdd(img)  # 单个图片
        img1 = torch.unsqueeze(img1, dim=0)

        # 测试
        # 打开文件夹测试
        num, pred = infer_fcdd(img1, self.model_fcdd)

        return num


def main(root=r'D:\MyCodes\pythonProject\coated_tongue_color\datas\data3\category\4\31.png'):
    image = Image.open(root)
    image = np.array(image)

    obj = DeploySZColor()

    return obj.run(image)


if __name__ == "__main__":
    main()
    # root = r'D:\MyCodes\pythonProject\coated_tongue_color\datas\data1&data2\category'
    # num = 0
    # for dir in os.listdir(root):
    #     for filename in tqdm(os.listdir(os.path.join(root, dir))):
    #         pred = main(os.path.join(root, dir, filename))
    #         if pred == 1:
    #             num = num + 1
    #             print(dir, filename, pred)
    # print(num)
