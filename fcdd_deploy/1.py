# 引入各个包
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


def proprecess_fcdd(data):
    test_transform = [
        transforms.Resize((248)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.435, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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
    if red_ascores < 0.074:
        pred = 0
    else:
        pred = 1
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

        print("--- load model sz_color successfully ---")

    def run(self, img0: np.ndarray):
        # 单张图片的格式统一为np.ndarray：[h,w,3]
        img = Image.fromarray(img0).convert("RGB")

        # 图片预处理
        img1 = proprecess_fcdd(img)  # 单个图片
        img1 = torch.unsqueeze(img1, dim=0)

        # 测试
        # 打开文件夹测试
        num, pred = infer_fcdd(img1, self.model_fcdd)

        return num


def main():
    image = Image.open("./6.png")
    image = np.array(image)

    obj = DeploySZColor()

    time1 = time.time()
    for _ in range(1):
        obj.run(image)
    print("time:", (time.time() - time1) / 1)


if __name__ == "__main__":
    main()
