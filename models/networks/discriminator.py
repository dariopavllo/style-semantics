"""
Adapted from: https://github.com/NVlabs/SPADE/blob/c0d50e6c4c106c0e88974f445f435b8ba5f4ccf6/models/networks/discriminator.py
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.normalization import SentenceSemanticAttention
import util.util as util


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        assert opt.num_D <= 2, 'Max 2 discriminators supported'
        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input, mask, attr, sentence, sentence_mask, return_attentions=False):
        result = []
        all_attentions = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        downsample = None
        for name, D in self.named_children():
            if return_attentions:
                out, attentions = D(input, mask, attr, sentence, sentence_mask, downsample, return_attentions=True)
                all_attentions.append(attentions)
            else:
                out = D(input, mask, attr, sentence, sentence_mask, downsample)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            downsample = self.downsample

        if return_attentions:
            return result, all_attentions
        else:
            return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        
        self.emb = nn.Conv2d(opt.label_nc, opt.mask_emb_dim, 1, bias=False)
        if self.opt.embed_attributes:
            attr_dim = opt.attr_emb_dim
            self.attr_emb = nn.Conv2d(opt.attr_nc, opt.attr_emb_dim, 1)
        else:
            attr_dim = 0
            self.attr_emb = None
            
        if self.opt.embed_captions:
            sent_dim = opt.attr_emb_dim # Same as attributes
            self.text_emb = SentenceSemanticAttention(opt.label_nc, query_dim=128, output_dim=sent_dim,
                                                      num_attention_heads=opt.attention_heads)
        else:
            sent_dim = 0
            self.text_emb = None
            
        sequence = [[nn.Conv2d(opt.output_nc + opt.mask_emb_dim + attr_dim + sent_dim, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        
        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))


    def forward(self, input, mask, attr, sentence, sentence_mask, downsample=None, return_attentions=False):
        m_orig = mask
        mask = self.emb(m_orig)
        
        cat_list = [input, mask]
        
        if self.opt.embed_attributes:
            attr = self.attr_emb(attr)
            cat_list.append(attr)
        
        if self.opt.embed_captions:
            if return_attentions:
                text, attentions = self.text_emb(m_orig, sentence, sentence_mask, return_attentions=True)
            else:
                text = self.text_emb(m_orig, sentence, sentence_mask)
            cat_list.append(text)
        
        input = torch.cat(cat_list, dim=1)
        if downsample is not None:
            input = downsample(input)
        results = [input]
        for submodel in self.children():
            if type(submodel) == nn.Embedding or submodel == self.emb or submodel == self.attr_emb or submodel == self.text_emb:
                continue
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            if return_attentions:
                return results[1:], attentions
            else:
                return results[1:]
        else:
            if return_attentions:
                return results[-1], attentions
            else:
                return results[-1]
