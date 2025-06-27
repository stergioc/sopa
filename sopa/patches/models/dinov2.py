import torch
from torch import nn


class DINOv2Features(nn.Module):
    def __init__(self, model="dinov2_vitl14"):
        super().__init__()
        self.dinov2_vitl14 = torch.hub.load("facebookresearch/dinov2", model)
        self.t = transforms.Compose([transforms.ToTensor()])

    def forward(self, x):
        feats = self.dinov2_vitl14.forward_features(self.t(x))
        return feats["x_norm_clstoken"]
