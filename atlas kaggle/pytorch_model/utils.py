import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
import torch
import time

labels = {
    '0' : 'Nucleoplasm',
    '1' : 'Nuclear membrane',
    '2' : 'Nucleoli',
    '3' : 'Nucleoli fibrillar center',
    '4' : 'Nuclear speckles',
    '5' : 'Nuclear bodies',
    '6' : 'Endoplasmic reticulum',
    '7' : 'Golgi apparatus',
    '8' : 'Peroxisomes',
    '9' : 'Endosomes',
    '10' : 'Lysosomes',
    '11' : 'Intermediate filaments',
    '12' : 'Actin filaments',
    '13' : 'Focal adhesion sites',
    '14' : 'Microtubules',
    '15' : 'Microtubule ends',
    '16' : 'Cytokinetic bridge',
    '17' : 'Mitotic spindle',
    '18' : 'Microtubule organizing center',
    '19' : 'Centrosome',
    '20' : 'Lipid droplets',
    '21' : 'Plasma membrane',
    '22' : 'Cell junctions',
    '23' : 'Mitochondria',
    '24' : 'Aggresome',
    '25' : 'Cytosol',
    '26' : 'Cytoplasmic bodies',
    '27' : 'Rods & rings',
}

channels = {
    0: 'Microtubules',
    1: 'Nucleus',
    2: 'Protein',
    3: 'Endoplasmic reticulum'
}

def download_data(names,
                project_folder='/home/vaden4d/Documents/kaggles/proteins/',
                folder='train',
                f='png'):
    data = []
    for name in names:
        full_name = os.path.join(project_folder, folder, name)
        image_red = np.array(Image.open(full_name + '_red.' + f))
        image_blue = np.array(Image.open(full_name + '_blue.' + f))
        image_green = np.array(Image.open(full_name + '_green.' + f))
        image_yellow= np.array(Image.open(full_name + '_yellow.' + f))
        image = np.dstack((image_red, image_blue, image_green, image_yellow))
        data.append(image)
    data = np.array(data)
    return data

def generate_batch(metadata, batch_size=3):
    n_batches = np.ceil(metadata.shape[0] / batch_size).astype(int)
    for i in range(n_batches):
        labels = metadata.iloc[i*(batch_size): (i+1)*batch_size].iloc[:, 1:].values
        names = metadata.iloc[i*(batch_size): (i+1)*batch_size].id
        yield download_data(names), labels

def compute_padding(input_shape, kernel_shape, strides_shape, dilation_shape):
    '''Compute optimal padding'''
    padding_shape = []
    for i, k, s, d in zip(input_shape, kernel_shape, strides_shape, dilation_shape):
        padding_shape.append(max((i - s*(i + 1) + d*(k - 1) + 1)//2, 0))
    padding_shape = tuple(padding_shape)
    return padding_shape

def macro_f1(y_pred, y_true):

    eps = 1e-8

    tp = (y_pred * y_true).sum(dim=0)
    tn = ((1 - y_pred) * (1 - y_true)).sum(dim=0)
    fp = (y_pred * (1 - y_true)).sum(dim=0)
    fn = ((1 - y_pred) * y_true).sum(dim=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = (2 * precision * recall) / (precision + recall + eps)

    return f1.mean()

class FLWithLogits(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, alpha, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()

class FL(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, alpha, logits, target):
        if not (target.size() == logits.size()):
            raise ValueError('Target size ({}) must be the same as input size ({})'
                             .format(target.size(), logits.size()))

        loss = target * F.sigmoid(-logits).pow(self.gamma) * \
        ((1.0 + (-logits.abs()).exp()).log() - logits.clamp(max=0)) + \
        (1 - target) * F.sigmoid(logits).pow(self.gamma) * \
        ((1.0 + (-logits.abs()).exp()).log() + logits.clamp(min=0))

        loss = loss @ alpha
        loss = loss.mean()

        return loss

if __name__ == '__main__':
    x = torch.Tensor([[14, -3], [-22, -11]])
    y = torch.Tensor([[1, 0], [1, 1]])
    print((y * (1 - F.sigmoid(x)).pow(2) * F.sigmoid(x).log() + \
    (1 - y) * F.sigmoid(x).pow(2) * (1 - F.sigmoid(x)).log()).sum(dim=1).mean())
    obj = FLWithLogits()
    start = time.time()
    alpha = torch.Tensor([0.5, 0.5])
    print(obj(alpha, x, y), time.time() - start)
    obj2 = FL()
    start = time.time()
    print(obj2(alpha, x, y), time.time() - start)
