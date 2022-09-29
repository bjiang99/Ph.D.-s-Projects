import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.gp = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        self.gn = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

#         self.load_state_dict(torch.load('results_jaccard/128_4096_0.1_0.999_200_512_500_model_TINY_2kernels_2lr1e3.pth', map_location='cpu'), strict=False)
        
    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out_p = self.gp(feature)
        out_n = self.gn(feature)
        return F.normalize(feature, dim=-1), F.normalize(out_p, dim=-1), F.normalize(out_n, dim=-1)