from .register import model_dict
import torch
import torch.nn as nn
import math

def get_backbone(backbone, pretrained=True):
    if backbone not in model_dict:
        raise ValueError("Backbone not supported")
    network = model_dict[backbone](pretrained=pretrained)
    return network

class MaskLayer(nn.Module):
    def __init__(self, d_in, d_out, masks, bias=True):
        super(MaskLayer, self).__init__()
        self.register_buffer('masks', masks)
        self.weight = nn.Parameter(torch.normal(0, 0.01, size=(d_out, d_in)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.bias = None
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, x):
        weight = self.weight * self.masks
        logits = torch.matmul(x, weight.T)
        if self.bias is not None:
            logits = logits + self.bias.unsqueeze(0)
        return logits

class Classifier(nn.Module):
    def __init__(self, backbone, num_classes, pretrained=True):
        super(Classifier, self).__init__()
        self.backbone = get_backbone(backbone, pretrained)
        num_features = self.backbone.num_features
        
        self.num_classes = num_classes
        self.num_features = num_features
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x, get_fea=False):
        fea = self.backbone(x)
        logits = self.fc(fea)
        if get_fea:
            return logits, fea
        else:
            return logits

class MaskedClassifier(nn.Module):
    def __init__(self, backbone, num_classes, masks, pretrained=True):
        super(MaskedClassifier, self).__init__()
        self.backbone = get_backbone(backbone, pretrained)
        num_features = self.backbone.num_features
        
        self.num_classes = num_classes
        self.num_features = num_features
        self.fc = MaskLayer(num_features, num_classes, masks)

    def update_mask(self, masks):
        self.fc.masks = masks

    def forward(self, x, get_fea=False):
        fea = self.backbone(x)
        logits = self.fc(fea)
        if get_fea:
            return logits, fea
        else:
            return logits

