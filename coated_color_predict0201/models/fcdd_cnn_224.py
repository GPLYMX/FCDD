import os.path as pt
import torch.nn as nn
import torchvision
#from algo_deploy.sz_color.models.bases import FCDDNet
from models.bases import FCDDNet
from torch.hub import load_state_dict_from_url

class FCDD_CNN224_VGG(FCDDNet):
    # VGG_11BN based net with most of the VGG layers having weights pretrained on the ImageNet classification task.
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, 'VGG net is only supported with bias atm!'
        state_dict = load_state_dict_from_url(
            torchvision.models.vgg.model_urls['vgg11_bn'],
            model_dir=pt.join(pt.dirname(__file__), '..', '..', '..', 'data', 'models')
        )
        features_state_dict = {k[9:]: v for k, v in state_dict.items() if k.startswith('features')}

        self.features = nn.Sequential(
            self._create_conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            self._create_conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            self._create_conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            self._create_conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            # Frozen version freezes up to here
            self._create_conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            self._create_conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # CUT
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.features.load_state_dict(features_state_dict)
        self.features = self.features[:-8]

        self.conv_final = self._create_conv2d(512, 1, 1)

    def forward(self, x, ad=True):
        x = self.features(x)

        if ad:
            x = self.conv_final(x)

        return x


class FCDD_CNN224_VGG_F(FCDD_CNN224_VGG):
    # VGG_11BN based net with most of the VGG layers having weights pretrained on the ImageNet classification task.
    # Additionally, these weights get frozen, i.e., the weights will not get updated during training.
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        for m in self.features[:15]:
            for p in m.parameters():
                p.requires_grad = False