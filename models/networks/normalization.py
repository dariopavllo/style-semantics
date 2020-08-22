"""
Adapted from: https://github.com/NVlabs/SPADE/blob/c0d50e6c4c106c0e88974f445f435b8ba5f4ccf6/models/networks/normalization.py
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm
import math

# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, mask_emb_dim, attr_nc, attr_emb_dim):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        
        if attr_nc == 0:
            attr_emb_dim = 0
            self.has_attr = False
        else:
            self.has_attr = True

        pw = ks // 2
        self.emb = nn.Conv2d(label_nc, mask_emb_dim, 1, bias=False)
        if self.has_attr:
            self.emb_attr = nn.Conv2d(attr_nc, attr_emb_dim, 1)
            
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(mask_emb_dim + attr_emb_dim, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, attr):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        ratio = segmap.shape[2] // x.shape[2]
        segmap = segmap[:, :, ::ratio, ::ratio]
        emb = self.emb(segmap)

        if self.has_attr:
            attr = attr[:, :, ::ratio, ::ratio]
            attr = self.emb_attr(attr)
        if self.has_attr:
            emb = torch.cat((emb, attr), dim=1)
        
        actv = self.mlp_shared(emb)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
    
class SPADEAvg(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, mask_emb_dim, attr_nc, attr_emb_dim):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        
        if attr_nc == 0:
            attr_emb_dim = 0
            self.has_attr = False
        else:
            self.has_attr = True

        pw = ks // 2
        self.emb = nn.Conv2d(label_nc, mask_emb_dim, 1, bias=False)
        if self.has_attr:
            self.emb_attr = nn.Conv2d(attr_nc, attr_emb_dim, 1, bias=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(mask_emb_dim + attr_emb_dim, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        
        self.mlp_shared_avg = nn.Sequential(
            nn.Conv2d(mask_emb_dim + attr_emb_dim, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma_avg = nn.Conv2d(nhidden, norm_nc, kernel_size=1, padding=0, bias=False)
        self.mlp_beta_avg = nn.Conv2d(nhidden, norm_nc, kernel_size=1, padding=0, bias=False)

    def forward(self, x, segmap, attr, segmap_avg, attr_avg):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        ratio = segmap.shape[2] // x.shape[2]
        segmap = segmap[:, :, ::ratio, ::ratio]
        segmap_avg = segmap_avg[:, :, ::ratio*2, ::ratio*2] # Save memory
        emb = self.emb(segmap)
        emb_avg = self.emb(segmap_avg)

        if self.has_attr:
            attr = attr[:, :, ::ratio, ::ratio]
            attr = self.emb_attr(attr)

            attr_avg = attr_avg[:, :, ::ratio*2, ::ratio*2] # Save memory
            attr_avg = self.emb_attr(attr_avg)
        if self.has_attr:
            emb = torch.cat((emb, attr), dim=1)
            emb_avg = torch.cat((emb_avg, attr_avg), dim=1)
        
        actv = self.mlp_shared(emb)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        
        actv_avg = self.mlp_shared_avg(emb_avg)
        actv_avg = torch.mean(actv_avg, dim=(2, 3), keepdim=True)
        gamma_avg = self.mlp_gamma_avg(actv_avg)
        beta_avg = self.mlp_beta_avg(actv_avg)

        # apply scale and bias
        out = normalized * (1 + gamma + gamma_avg) + beta + beta_avg

        return out
    
class SentenceSemanticAttention(nn.Module):
    def __init__(self, n_embeddings, query_dim=768, output_dim=768, hidden_size=768, num_attention_heads=6):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = 64 # Hardcoded
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        
        self.output = nn.Linear(self.all_head_size, output_dim)
        
        self.query = nn.Parameter(torch.randn(n_embeddings, self.all_head_size))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, sentence, attention_mask, return_attentions=False):
        """
        Based on the multihead attention implementation by huggingface transformers:
        https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py
        """
        
        # x is a one-hot vector of shape (N, n_embeddings, h, w)
        
        mixed_key_layer = self.key(sentence)
        mixed_value_layer = self.value(sentence)
        mixed_query_layer = self.query
        
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        query_layer = self.transpose_for_scores(mixed_query_layer.unsqueeze(0))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Prepare attention mask
        attention_mask = (1 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0
        
        # Apply the attention mask
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        contextual_embeddings = self.output(context_layer) # (N, n_embeddings, output_dim)
        
        x_shape = x.shape
        x = x.contiguous().view(x.shape[0], x.shape[1], -1) # Flatten pixels
        
        res = torch.matmul(contextual_embeddings.transpose(-1, -2), x) # (N, output_dim, n_pixels)
        res = res.view(res.shape[0], res.shape[1], x_shape[2], x_shape[3])
        
        if return_attentions:
            return res, attention_probs
        else:
            return res
    
    
class SPADEAttention(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, mask_emb_dim, output_emb_dim, num_attention_heads):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        
        self.emb = nn.Conv2d(label_nc, mask_emb_dim, 1, bias=False)
        self.emb_attr = SentenceSemanticAttention(label_nc, query_dim=128, output_dim=output_emb_dim,
                                                  num_attention_heads=num_attention_heads)
            
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(mask_emb_dim + output_emb_dim, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, sentence, sentence_len):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        ratio = segmap.shape[2] // x.shape[2]

        segmap = segmap[:, :, ::ratio, ::ratio]
        emb = self.emb(segmap)

        attr = self.emb_attr(segmap, sentence, sentence_len)
        emb = torch.cat((emb, attr), dim=1)
        
        actv = self.mlp_shared(emb)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
    
