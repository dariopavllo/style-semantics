"""
Adapted from: https://github.com/NVlabs/SPADE/blob/c0d50e6c4c106c0e88974f445f435b8ba5f4ccf6/models/networks/architecture.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE, SPADEAvg, SPADEAttention, get_nonspade_norm_layer


class SBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        
        if opt.embed_attributes:
            attr_nc = opt.attr_nc
        else:
            attr_nc = 0
        
        if opt.embed_captions:
            class_inst = SPADEAttention
            extra_args = [opt.attr_emb_dim, opt.attention_heads]
        else:
            class_inst = SPADE
            extra_args = [attr_nc, opt.attr_emb_dim] # Can be zero
        
        self.norm_0 = class_inst(spade_config_str, fin, opt.label_nc, opt.mask_emb_dim, *extra_args)
        self.norm_1 = class_inst(spade_config_str, fmiddle, opt.label_nc, opt.mask_emb_dim, *extra_args)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.label_nc, opt.mask_emb_dim, attr_nc, opt.attr_emb_dim)

    def forward(self, x, seg, attr, sentence, sentence_mask):
        x_s = self.shortcut(x, seg, attr)

        if self.opt.embed_captions:
            args = [sentence, sentence_mask]
        else:
            args = [attr] # This can as well be None
        dx = self.conv_0(self.actvn(self.norm_0(x, seg, *args)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, *args)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg, attr):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, attr))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1, inplace=True)


    
class SAvgBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        
        if opt.embed_attributes:
            attr_nc = opt.attr_nc
        else:
            attr_nc = 0

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADEAvg(spade_config_str, fin, opt.label_nc, opt.mask_emb_dim, attr_nc, opt.attr_emb_dim)
        self.norm_1 = SPADEAvg(spade_config_str, fmiddle, opt.label_nc, opt.mask_emb_dim, attr_nc, opt.attr_emb_dim)
        if self.learned_shortcut:
            self.norm_s = SPADEAvg(spade_config_str, fin, opt.label_nc, opt.mask_emb_dim, attr_nc, opt.attr_emb_dim)

    def forward(self, x, seg, attr, seg_avg, attr_avg):
        x_s = self.shortcut(x, seg, attr, seg_avg, attr_avg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg, attr, seg_avg, attr_avg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, attr, seg_avg, attr_avg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg, attr, seg_avg, attr_avg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, attr, seg_avg, attr_avg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1, inplace=True)
    
    
    
class NonSBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        
        norm_layer = get_nonspade_norm_layer(opt, 'spectralsync_batch')

        # create conv layers
        self.conv_0 = norm_layer(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = norm_layer(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

    def forward(self, x):
        x_s = self.shortcut(x)

        dx = self.actvn(self.conv_0(x))
        dx = self.actvn(self.conv_1(dx))

        out = x_s + dx

        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1, inplace=True)


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out