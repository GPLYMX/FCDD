# -*- coding: utf-8 -*-
# @Author : guopeng
# @Version : 2.1
# @Time : 2022/11/15 14:42:25
import time, torch, os
import torch.nn as nn
from torchvision.models import resnet152
from torchvision import transforms
from PIL import Image
import numpy as np

import fcdd
from quantification import coated_tongue_color_quantization
from quantification import revise


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.shape = 0

    def forward(self, x):
        self.shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, self.shape)


class CoatedTongueColorModel(nn.Module):

    def __init__(self):
        super(CoatedTongueColorModel, self).__init__()
        self.trained_model = resnet152(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(2048, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, 4),
                                    )
        # self.trained_model2 = vgg16(pretrained=True)  # .to(device)
        self.model2 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    # 测试一下输出维度[b, 512, 1, 1]
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.Dropout(p=0.4),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.4),
                                    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.4),
                                    nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.AdaptiveAvgPool2d(output_size=(7, 7)),
                                    Flatten(),
                                    nn.Linear(12544, 2000),
                                    nn.ReLU(),
                                    nn.Linear(2000, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 4)
                                    )
        self.model3 = nn.Sequential(
            nn.Linear(8, 4)
        )

    def forward(self, input1):
        x1 = self.model1(input1)
        x2 = self.model2(input1)
        output1 = torch.cat([x1, x2], dim=1)
        output1 = self.model3(output1)
        return output1


def preprocess_img(original_img):
    """
    :param original_img: RGB（PIL）
    :return:torch.tensor格式，shape=[b,c,h,w]
    """
    resize = 256
    preprocessing = transforms.Compose([
        # lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.435, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    processed_img = preprocessing(original_img)
    processed_img = processed_img.unsqueeze(0)
    return processed_img


def infer(input_img, input_model1, model_fcdd):
    """
    说明1、img格式为RGB（PIL）或者BGR（Opencv），需要备注格式；可能单个图片，也可能是列表，请备注
    说明2：提供模型预测值和最终标签的字典映射关系
    """
    index2label = {0: "白", 1: "淡黄", 2: "黄", 3: "焦黄", 4: "灰黑"}  # 模型预测值和标签的字典映射关系，类似这个

    quantized_value = coated_tongue_color_quantization(input_img)

    pred = model_fcdd.run(input_img)
    if pred == int(1):
        index = int(4)
    else:
        input_img = Image.fromarray(input_img).convert("RGB")
        input_img = preprocess_img(input_img)
        index = int(input_model1(input_img).argmax(dim=1))  # 模型可能输出不止一个字段值时，备注各个字段含义
    label = index2label[index]  # 这个必选项

    # 修正量化值
    quantized_value = revise(quantized_value, index)

    return label, quantized_value


class DeployTaise(object):
    def __init__(self):
        device = torch.device("cpu")  # 限定使用cpu
        self.model1 = CoatedTongueColorModel()
        path_tmp = os.path.join(os.path.dirname(__file__), "coated_tongue_color_model221109.mdl")
        self.model1.load_state_dict(torch.load(path_tmp, map_location=device))
        self.model1.eval()
        self.model1.to(device=device)

        self.model_fcdd = fcdd.DeploySZColor()

    def run(self, img: np.ndarray):
        # img = Image.fromarray(img).convert("RGB")
        label, quantized_value = infer(img, self.model1, self.model_fcdd)
        return [label, quantized_value]


if __name__ == "__main__":
    # 加载图片为RGB格式
    # picture_root = r'D:\MyCodes\pythonProject\coated_tongue_color\datas\data3\category\4\306.png'
    picture_root = r'D:\MyCodes\pythonProject\siamese\datas\test\0\26.png'
    img = Image.open(picture_root).convert('RGB')

    ts1 = time.time()
    obj = DeployTaise()
    ts2 = time.time()
    print("ts1: {}".format(ts2 - ts1))

    for _ in range(10):
        pred = obj.run(np.array(img))
        print(pred)
    ts = time.time() - ts2
    print("ts1: {}".format(ts / 10))
    # pass
    #
    # obj = DeployLaonen()
    #
    # def proc(root):
    #     img = Image.open(root).convert('RGB')
    #     pred = obj.run(np.array(img))
    #     return pred
    # root = r'D:\MyCodes\pythonProject\coated_tongue_color\datas\data1&data2\category\0'
    # for file in tqdm(os.listdir(root)):
    #     a = proc(os.path.join(root, file))
    #     if a != '白':
    #         print(file, a)

    # root = r'D:\MyCodes\pythonProject\coated_tongue_color\datas\data1&data2\category\1'
    # for file in tqdm(os.listdir(root)):
    #     a = proc(os.path.join(root, file))
    #     if a != '淡黄':
    #         print(file, a)
    #
    # root = r'D:\MyCodes\pythonProject\coated_tongue_color\datas\data1&data2\category\2'
    # for file in tqdm(os.listdir(root)):
    #     a = proc(os.path.join(root, file))
    #     if a != '黄':
    #         print(file, a)
    #
    # root = r'D:\MyCodes\pythonProject\coated_tongue_color\datas\data1&data2\category\3'
    # for file in tqdm(os.listdir(root)):
    #     a = proc(os.path.join(root, file))
    #     if a != '焦黄':
    #         print(file, a)
    # root = r'D:\MyCodes\pythonProject\coated_tongue_color\datas\data3\category\4'
    # for file in tqdm(os.listdir(root)):
    #     a = proc(os.path.join(root, file))
    #     if a != '灰黑':
    #         print(file, a)
