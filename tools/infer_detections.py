#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
import numpy as np

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    parser.add_argument(
        '--use-vg3k',
        dest='use_vg3k',
        help='use Visual Genome 3k classes (instead of COCO 80 classes)',
        action='store_true'
    )
    parser.add_argument(
        '--thresh',
        default=0.0,
        type=float,
        help='filter predictions using threshold (set to 0 to save everything)',
    )
    parser.add_argument(
        '--filter-classes',
        action='store_true',
        help='filter classes using hardcoded class list (reduces size)',
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

all_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 72, 74, 75, 76, 77, 94, 141, 170, 196, 211, 213, 217, 223, 227, 232, 246, 247, 259, 263, 269, 270, 277, 282, 285, 290, 307, 341, 347, 360, 381, 394, 402, 406, 412, 416, 428, 430, 436, 444, 445, 468, 471, 475, 488, 494, 497, 510, 532, 550, 561, 573, 591, 602, 606, 632, 644, 663, 675, 680, 696, 703, 715, 728, 735, 736, 753, 766, 771, 779, 814, 819, 821, 870, 922, 935, 939, 960, 972, 975, 976, 1011, 1041, 1066, 1067, 1076, 1083, 1120, 1140, 1147, 1161, 1185, 1200, 1244, 1246, 1250, 1258, 1268, 1276, 1308, 1314, 1318, 1333, 1342, 1345, 1354, 1365, 1378, 1390, 1397, 1398, 1404, 1414, 1470, 1472, 1474, 1476, 1499, 1520, 1526, 1535, 1539, 1541, 1560, 1575, 1579, 1586, 1587, 1596, 1624, 1643, 1651, 1653, 1654, 1685, 1696, 1714, 1728, 1745, 1750, 1769, 1775, 1777, 1783, 1793, 1802, 1810, 1825, 1871, 1879, 1916, 1924, 1939, 1946, 1950, 1959, 1972, 1982, 1983, 1992, 2010, 2011, 2012, 2029, 2030, 2061, 2064, 2090, 2113, 2124, 2135, 2186, 2195, 2199, 2206, 2207, 2210, 2212, 2213, 2215, 2223, 2225, 2248, 2249, 2271, 2283, 2299, 2326, 2341, 2352, 2354, 2366, 2375, 2384, 2416, 2424, 2426, 2444, 2451, 2467, 2472, 2488, 2493, 2544, 2550, 2582, 2588, 2611, 2618, 2637, 2651, 2659, 2666, 2686, 2695, 2698, 2720, 2728, 2738, 2753, 2759, 2762, 2769, 2772, 2774, 2783, 2784, 2795, 2797, 2834, 2849, 2862, 2884, 2896, 2897, 2898, 2910, 2927, 2947, 2953, 2954, 2956, 2958, 2963, 2964, 2967, 2988]


def convert(cls_boxes, cls_segms):
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, classes

def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = (
        dummy_datasets.get_vg3k_dataset()
        if args.use_vg3k else dummy_datasets.get_coco_dataset())

    if os.path.isdir(args.im_or_folder):
        im_list = sorted(glob.iglob(args.im_or_folder + '/*.' + args.image_ext))
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.npz')
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        if im is None:
            logger.info('Unable to read image, skipping.')
            continue
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        boxes, segms, classes = convert(cls_boxes, cls_segms)
        classes = np.array(classes, dtype=np.uint16)
        resolution = segms[0]['size']
        segms = np.array([x['counts'] for x in segms]) # Run-length encoding
        
        valid = boxes[:, 4] >= args.thresh
        if args.filter_classes:
            valid &= np.isin(classes, all_classes)
            
        boxes = boxes[valid].copy()
        classes = classes[valid].copy()
        segms = segms[valid].copy()
        
        output_name = os.path.basename(im_name)
        np.savez(args.output_dir + '/' + output_name, boxes=boxes, segments=segms, classes=classes, resolution=resolution)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
