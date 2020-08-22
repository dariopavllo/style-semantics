"""
Adapted from: https://github.com/NVlabs/SPADE/blob/c0d50e6c4c106c0e88974f445f435b8ba5f4ccf6/models/networks/generator.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.architecture import SBlock, SAvgBlock, NonSBlock
from models.networks.normalization import SentenceSemanticAttention

# This class serves as the background generator in the two-step model or the generator in the one-step model
class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)
        

        self.mask_emb = nn.Conv2d(opt.label_nc, opt.mask_emb_dim, 1, bias=False)
        if self.opt.embed_attributes:
            attr_emb_dim = opt.attr_emb_dim
            self.attr_emb = nn.Conv2d(opt.attr_nc, attr_emb_dim, 1)
        else:
            attr_emb_dim = 0
            
        if self.opt.embed_captions:
            text_emb_dim = opt.attr_emb_dim # Same as attributes
            self.text_emb = SentenceSemanticAttention(opt.label_nc, query_dim=128, output_dim=text_emb_dim,
                                                      num_attention_heads=opt.attention_heads)
        else:
            text_emb_dim = 0

        self.fc = nn.Conv2d(opt.mask_emb_dim + attr_emb_dim + text_emb_dim, 16 * nf, 3, padding=1)

        # The S_avg branch is missing when we condition the two-step model on text.
        # Text alone is as informative as the global-average-pooled foreground mask,
        # so we can reduce the computational cost by removing that branch.
        if self.opt.two_step_model and not self.opt.embed_captions:
            self.fc_avg = nn.Conv2d(opt.mask_emb_dim + attr_emb_dim, 16 * nf, 3, padding=1)
            block_inst = SAvgBlock
        else:
            block_inst = SBlock
            

        self.head_0 = block_inst(16 * nf, 16 * nf, opt)

        self.G_middle_0 = block_inst(16 * nf, 16 * nf, opt)
        if not self.opt.two_step_model:
            self.G_middle_1 = block_inst(16 * nf, 16 * nf, opt)

        self.up_0 = block_inst(16 * nf, 8 * nf, opt)
        self.up_1 = block_inst(8 * nf, 4 * nf, opt)
        self.up_2 = block_inst(4 * nf, 2 * nf, opt)
        self.up_3 = block_inst(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = block_inst(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, attr, input_avg, attr_avg, sentence, sentence_mask):
        seg = input
        seg_avg = input_avg
        
        ratio = seg.shape[2] // self.sh
        seg_ds = seg[:, :, ::ratio, ::ratio] # Downsample
        x = self.mask_emb(seg_ds)
        
        if self.opt.embed_attributes:
            attr_ = attr[:, :, ::ratio, ::ratio] # Downsample
            attr_ = self.attr_emb(attr_)
            x = torch.cat((x, attr_), dim=1)
            
        if self.opt.embed_captions:
            text = self.text_emb(seg_ds, sentence, sentence_mask)
            x = torch.cat((x, text), dim=1)
            
        x = self.fc(x)

        if self.opt.two_step_model and not self.opt.embed_captions:
            x_avg = seg_avg[:, :, ::ratio, ::ratio] # Downsample
            x_avg = self.mask_emb(x_avg)
            if self.opt.embed_attributes:
                attr_avg_ = attr_avg[:, :, ::ratio, ::ratio] # Downsample
                attr_avg_ = self.attr_emb(attr_avg_)
                x_avg = torch.cat((x_avg, attr_avg_), dim=1)
            x_avg = torch.mean(x_avg, dim=(2, 3), keepdim=True) # Global average pooling (remove positional info)
            x_avg = self.fc_avg(x_avg)
            x = x + x_avg
            block_args = [seg, attr, seg_avg, attr_avg]
        else:
            block_args = [seg, attr, sentence, sentence_mask]
            
        x = self.head_0(x, *block_args)

        x = self.up(x)
        x = self.G_middle_0(x, *block_args)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            
        if not self.opt.two_step_model:
            x = self.G_middle_1(x, *block_args)

        x = self.up(x)
        x = self.up_0(x, *block_args)
        x = self.up(x)
        x = self.up_1(x, *block_args)
        x = self.up(x)
        x = self.up_2(x, *block_args)
        x = self.up(x)
        x = self.up_3(x, *block_args)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, *block_args)

        x = self.conv_img(F.leaky_relu(x, 2e-1, inplace=True)).tanh_()

        return x
    
    
class SPADEfgGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.conv1 = nn.Conv2d(3, nf, 5, padding=2)
        
        
        self.down_1 = NonSBlock(1*nf, 2*nf, opt)
        self.down_2 = NonSBlock(2*nf, 3*nf, opt)
        self.down_3 = NonSBlock(3*nf, 4*nf, opt)
        self.down_4 = NonSBlock(4*nf, 16*nf, opt)
        
        self.middle_1 = SBlock(16*nf, 16*nf, opt)
        
        self.up_1 = SBlock(16*nf, 8*nf, opt)
        self.up_2 = SBlock(8*nf, 4*nf, opt)
        self.up_3 = SBlock(4*nf, 2*nf, opt)
        self.up_4 = SBlock(2*nf, 1*nf, opt)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)
        self.conv_alpha = nn.Conv2d(nf, 1, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)
        self.down = nn.AvgPool2d(2)

    def forward(self, input_img, input_seg, input_attr, sentence, sentence_mask):
        x = input_img
            
        x = self.conv1(x)
        
        x = self.down_1(x)
        x = self.down(x)
        
        x = self.down_2(x)
        x = self.down(x)
        
        x = self.down_3(x)
        x = self.down(x)
        
        x = self.down_4(x)
        x = self.down(x)
        
        x = self.middle_1(x, input_seg, input_attr, sentence, sentence_mask)
        
        x = self.up(x)
        x = self.up_1(x, input_seg, input_attr, sentence, sentence_mask)
        
        x = self.up(x)
        x = self.up_2(x, input_seg, input_attr, sentence, sentence_mask)
        
        x = self.up(x)
        x = self.up_3(x, input_seg, input_attr, sentence, sentence_mask)
        
        x = self.up(x)
        x = self.up_4(x, input_seg, input_attr, sentence, sentence_mask)
        
        arg = x
        x = self.conv_img(arg).tanh_()
        alpha_logits = self.conv_alpha(arg)
        
        alpha = torch.sigmoid(alpha_logits)
        x = input_img * (1 - alpha) + x*alpha
        
        return x, alpha_logits
    
