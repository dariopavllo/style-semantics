"""
Batched FID evaluation script.
The functions to calculate statistics have been borrowed from pytorch-fid:
https://github.com/mseitzer/pytorch-fid
"""

import os
from collections import OrderedDict

import numpy as np
from scipy import linalg
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import torch.nn as nn

import torch
from util.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    print('Warning: tqdm not found. Install it to see progress bar.')
    def tqdm(x): return x

opt = TestOptions().parse()
opt.concat_all_captions = True # For FID evaluation, we concatenate all captions

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
if len(opt.gpu_ids) > 1:
    model = nn.DataParallel(model, device_ids=opt.gpu_ids)
model.eval()

block_dim = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[block_dim]
inception_model = InceptionV3([block_idx])
if len(opt.gpu_ids) > 1:
    inception_model = nn.DataParallel(inception_model, device_ids=opt.gpu_ids)
if len(opt.gpu_ids) > 0:
    inception_model = inception_model.cuda()
inception_model.eval()

# test
emb_arr_fake = []
emb_arr_real = []
for i, data_i in enumerate(tqdm(dataloader)):
    if i * opt.batchSize >= opt.how_many:
        break

    generated_bg, generated, mask = model(data_i, mode='inference')
    with torch.no_grad():
        batch_inception = torch.cat((generated, data_i['image'].cuda()), dim=0)/2 + 0.5 # (map to 0, 1)
        pred = inception_model(batch_inception)[0]
        
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            
        emb_fake, emb_real = torch.chunk(pred, 2, dim=0)
        emb_arr_fake.append(emb_fake.cpu().data.numpy().reshape(-1, block_dim))
        emb_arr_real.append(emb_real.cpu().data.numpy().reshape(-1, block_dim))

emb_arr_fake = np.concatenate(emb_arr_fake, axis=0)
emb_arr_real = np.concatenate(emb_arr_real, axis=0)

def calculate_stats(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

m1, s1 = calculate_stats(emb_arr_fake)
m2, s2 = calculate_stats(emb_arr_real)

fid_value = calculate_frechet_distance(m1, s1, m2, s2)
print("FID: ", fid_value)