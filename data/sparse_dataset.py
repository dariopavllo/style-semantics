from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image, ExifTags
from data.image_folder import make_dataset
from collections import Counter
from scipy.ndimage import shift
import util.util as util
import os
import json
import numpy as np
import pycocotools.mask as mask_util
import torch
import re
import torch.nn.functional as F

from data.label_list import vg_remapped_classes, vg_labels
from data.label_list import stuff_classes, always_stuff_classes, thing_classes
from data.label_list import excluded_attributes, remapped_attributes

class SparseDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=281)

        return parser

    def initialize(self, opt):
        self.opt = opt
        self.debug_mode = False
        self.attribute_hook = None
        self.class_hook = None
        self.threshold = 0.2
        self.manual_caption_id = 0

        label_paths, image_paths, precomputed_captions_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        
        if opt.embed_captions and opt.use_precomputed_captions:
            util.natural_sort(precomputed_captions_paths)
            precomputed_captions_paths = precomputed_captions_paths[:opt.max_dataset_size]
        else:
            assert precomputed_captions_paths is None

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1[:-4], path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)
                
            if opt.embed_captions and opt.use_precomputed_captions:
                for path1, path3 in zip(label_paths, precomputed_captions_paths):
                    assert self.paths_match(path1[:-4], path3, cast_to_int=True), \
                        "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path3)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.precomputed_captions_paths = precomputed_captions_paths

        size = len(self.label_paths)
        self.dataset_size = size
        
        self.initialize_metadata(opt)
        
        if opt.embed_captions and not opt.use_precomputed_captions:
            # Import huggingface tokenizer
            print('Precomputed captions are not being used. Importing huggingface tokenizer library...')
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_paths(self, opt):
        # Must be implemented by subclass
        raise NotImplementedError
    
    def initialize_metadata(self, opt):
        
        self.vg_labels = vg_labels
        self.vg_labels_set = set(vg_labels)
        if opt.embed_attributes:
            # Load VG config
            print('Loading attribute json....')
            with open('datasets/vg/attributes.json', 'r') as f:
                self.vg_attributes = json.load(f)
            with open('datasets/vg/attribute_synsets.json', 'r') as f:
                self.vg_attribute_synsets = json.load(f)
            
            if self.coco_match_attributes:
                coco_resolutions = {}
                anno_file = 'instances_train2017.json' if opt.isTrain else 'instances_val2017.json'
                with open(os.path.join('datasets/coco/annotations', anno_file), 'r') as f:
                    coco_val_images = json.load(f)['images']
                    for im in coco_val_images:
                        coco_resolutions[im['id']] = (im['width'], im['height'])
            
            with open('datasets/vg/image_data.json', 'r') as f:
                image_data = json.load(f)
                self.indices = {}
                self.coco_to_vg = {}
                self.vg_to_coco = {}
                for i, img in enumerate(image_data):
                    self.indices[img['image_id']] = i
                    if img['coco_id'] is not None:
                        self.coco_to_vg[img['coco_id']] = img['image_id']
                        self.vg_to_coco[img['image_id']] = (img['coco_id'], img['width'], img['height'])

            self.attribute_counter = Counter()
            for img in self.vg_attributes:
                regions = img['attributes']
                for region in regions:
                    if 'attributes' in region:
                        attr_synsets = []
                        for x in region['attributes']:
                            x = x.lower().strip()
                            split = [y.strip() for y in re.split(' and |-and-|, and|, |,|\\+', x)]
                            for x in split:
                                if x in self.vg_attribute_synsets.keys():
                                    candidate = self.vg_attribute_synsets[x]
                                else:
                                    candidate = x
                                if candidate not in excluded_attributes:
                                    if candidate in remapped_attributes:
                                        candidate = remapped_attributes[candidate]
                                    attr_synsets.append(candidate)
                        attr_synsets = list(set(attr_synsets)) # Remove duplicates
                        for attr in attr_synsets:
                            self.attribute_counter[attr] += 1
                        region['attribute_synsets'] = attr_synsets

            # Select top-k attributes
            self.attr_map = {}
            self.attr_map_inv = {}
            for attr, freq in self.attribute_counter.most_common(self.opt.attr_nc):
                tgt = len(self.attr_map) + 1
                self.attr_map[attr] = tgt
                self.attr_map_inv[tgt] = attr
            
            # Select top-k classes
            all_synsets = []
            for img in self.vg_attributes:
                regions = img['attributes']
                for region in regions:
                    for synset in region['synsets']:
                        all_synsets.append(synset)

            vg3k_cls_names = all_synsets

            vg3k_cls_names_clean = [
                name[:-5].replace('_', ' ') for name in vg3k_cls_names]

            vg3k_clean2original = {
                n1: n2 for n1, n2 in zip(vg3k_cls_names_clean[::-1], vg3k_cls_names[::-1])}

            vg3k_original2clean = {v: k for k, v in vg3k_clean2original.items()}

            match_count = 0
            for img in self.vg_attributes:
                regions = img['attributes']
                region_db = {}
                for region in regions:
                    if len(region['synsets']) == 1:
                        synset = region['synsets'][0]
                        if synset in vg3k_original2clean:
                            region['clean_label'] = vg3k_original2clean[synset]
                            if region['clean_label'] not in region_db:
                                region_db_list = []
                                region_db[region['clean_label']] = region_db_list
                            else:
                                region_db_list = region_db.get(region['clean_label'])
                            region_db_list.append(region)
                img['region_db'] = region_db
                
                if self.coco_match_attributes:
                    # Update resolution
                    if img['image_id'] in self.vg_to_coco:
                        coco_id, w, h = self.vg_to_coco[img['image_id']]
                        if coco_id in coco_resolutions: # This is only true for the val set
                            coco_res = coco_resolutions[coco_id]
                            w_ratio = coco_res[0] / w
                            h_ratio = coco_res[1] / h
                            for r1 in region_db.values():
                                for region in r1:
                                    region['x'] = region['x'] * w_ratio
                                    region['w'] = region['w'] * w_ratio
                                    region['y'] = region['y'] * h_ratio
                                    region['h'] = region['h'] * h_ratio
                            match_count += 1
            print('Images matched between COCO and VG:', match_count)
            
        self.remapped_classes = vg_remapped_classes
        
        self.all_classes = stuff_classes + thing_classes
        self.remapped_idx = {k: pos+1 for pos, k in enumerate(self.all_classes)}
        self.remapped_idx_reverse = {v: k for k, v in self.remapped_idx.items()}
        
        # Starting from index 1 (0 is none)
        self.stuff_classes = np.arange(len(stuff_classes)) + 1
        self.thing_classes = np.arange(len(thing_classes)) + len(self.stuff_classes) + 1
        
        self.always_stuff = always_stuff_classes
        
        self.opt.bg_labels = self.stuff_classes
        self.opt.fg_labels = self.thing_classes

    def paths_match(self, path1, path2, cast_to_int=False):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        if cast_to_int:
            return int(filename1_without_ext) == int(filename2_without_ext)
        else:
            return filename1_without_ext == filename2_without_ext
    
    def make_mask(self, filename, res, num_attr_masks=5, debug=False):
        data = np.load(filename, encoding='latin1', allow_pickle=True)

        region_db = None
        if self.opt.embed_attributes and self.coco_match_attributes is not None:
            image_id = int(os.path.basename(filename).split('.')[0])
            if not self.coco_match_attributes:
                region_db = self.vg_attributes[self.indices[image_id]]['region_db']
            elif image_id in self.coco_to_vg:
                image_id = self.coco_to_vg[image_id]
                region_db = self.vg_attributes[self.indices[image_id]]['region_db']

        def parse(key):
            d = data[key]
            if len(d.shape) == 0:
                return d.item()
            else:
                return d

        boxes = parse('boxes')
        segms = parse('segments')
        classes = parse('classes')
        parsed_res = parse('resolution')

        out_mask = np.empty((res[1], res[0]), dtype=np.uint16)
        out_mask[:] = 0
        
        out_mask_bg = np.empty((res[1], res[0]), dtype=np.uint16)
        out_mask_bg[:] = 0
        
        out_mask_fg = np.empty((res[1], res[0]), dtype=np.uint16)
        out_mask_fg[:] = 0

        attr_masks = []
        for i in range(num_attr_masks):
            attr_mask = np.zeros((res[1], res[0]), dtype=np.uint16)
            attr_masks.append(attr_mask)
            
        attr_masks_bg = []
        for i in range(num_attr_masks):
            attr_mask_bg = np.zeros((res[1], res[0]), dtype=np.uint16)
            attr_masks_bg.append(attr_mask_bg)
            
        attr_masks_fg = []
        for i in range(num_attr_masks):
            attr_mask_fg = np.zeros((res[1], res[0]), dtype=np.uint16)
            attr_masks_fg.append(attr_mask_fg)
        
        def iou_mask(a, b):
            assert a.shape == b.shape
            intersection = np.sum(a & b)
            if intersection == 0:
                return 0
            union = np.sum(a | b)
            return intersection/union

        def iou_bb(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
            yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])

            interArea = max(0, xB - xA) * max(0, yB - yA)

            boxAArea = boxA[2] * boxA[3]
            boxBArea = boxB[2] * boxB[3]

            iou = interArea / (boxAArea + boxBArea - interArea)
            return iou
        
        def bb_intersect(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
            yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            return interArea > 1e-6
        
        def diff_mask(parent, child):
            assert parent.shape == child.shape
            intersection = np.sum(parent & child)
            if intersection == 0:
                return 0
            child_area = np.sum(child)
            return intersection/child_area
        
        thresh = self.threshold
        if boxes is not None:
            scores = boxes[:, -1]
            valid = scores > thresh
            original_classes = classes.copy()
            
            # Remap classes as needed
            for i in range(len(classes)):
                if classes[i] in self.remapped_classes:
                    classes[i] = self.remapped_classes[classes[i]]
            
            for i, cl in enumerate(classes):
                if cl not in self.remapped_idx:
                    valid[i] = False

            if segms is not None and len(segms) > 0:
                masks = mask_util.decode([{'counts': bytes(x), 'size': parsed_res} for x in segms])
                
                # Perform NMS
                valid_indices, = np.where(valid)
                valid_indices = valid_indices[np.argsort(-scores[valid_indices])]
                for j1, i1 in enumerate(valid_indices):
                    if not valid[i1]:
                        continue
                        
                    bb1 = boxes[i1, :4].copy()
                    bb1[2:] -= bb1[:2]
                    
                    for j2, i2 in enumerate(valid_indices):
                        if j2 > j1 and valid[i2]:
                            assert scores[i1] >= scores[i2], (scores[i1], scores[i2]) # Check if sorted
                            bb2 = boxes[i2, :4].copy()
                            bb2[2:] -= bb2[:2]
                            overlap_bb = iou_bb(bb1, bb2)
                            if overlap_bb > 1e-3:
                                m1 = masks[:, :, i1]
                                m2 = masks[:, :, i2]
                                overlap_mask = iou_mask(m1, m2)
                                if overlap_mask > 0.7:
                                    if scores[i1] > scores[i2]:
                                        valid[i2] = False

                classes = np.array(classes)[valid]
                original_classes = np.array(original_classes)[valid]
                masks = masks[:, :, valid]
                scores = boxes[valid, -1]
                boxes = boxes[valid, :4]
                assert masks.shape[:2] == out_mask.shape, (filename, masks.shape, out_mask.shape)

                mask_ref = np.zeros(masks.shape[:2], dtype=int)

                if debug:
                    mask_debug = np.empty(masks.shape[:2], dtype=object)
                    mask_debug[:] = None

                areas = np.sum(masks, axis=(0, 1))
                draw_indices = np.argsort(-areas)

                drawn_masks = []
                drawn_masks_indices = []
                drawn_masks_scores = []
                
                scene_graph = []
                render_queue = []
                
                for idx in draw_indices:
                    class_name = vg_labels[classes[idx]]
                    original_class_name = vg_labels[original_classes[idx]]
                    if self.class_hook is not None:
                        bool_mask = masks[:, :, idx].astype('bool')
                        new_class, delta_pos = self.class_hook(idx, class_name, bool_mask)
                        if new_class is None:
                            continue # Do not draw this object
                        classes[idx] = vg_labels.index(new_class)
                        class_name = new_class
                        
                    if classes[idx] in self.remapped_idx:
                        bool_mask = masks[:, :, idx].astype('bool')
                            
                        if self.class_hook is not None:
                            bool_mask = shift(bool_mask, delta_pos, order=0)
                            
                        if np.count_nonzero(bool_mask) == 0:
                            continue
                            
                        # Add node to scene graph
                        is_foreground = False
                        candidate = scene_graph
                        bb1 = boxes[idx, :4].copy()
                        bb1[2:] -= bb1[:2]
                        def add_to_scene_graph():
                            nonlocal candidate, is_foreground
                            added = False
                            for cur_node in candidate:
                                existing_mask = cur_node['mask']
                                bb2 = cur_node['bbox'].copy()
                                bb2[2:] -= bb2[:2]
                                if not bb_intersect(bb1, bb2):
                                    continue # Early out
                                overlap_diff = diff_mask(existing_mask, bool_mask)
                                if overlap_diff > 0.7:
                                    #assert not added
                                    candidate = cur_node['children']
                                    is_foreground = cur_node['is_foreground']
                                    add_to_scene_graph()
                                    added = True
                        
                        add_to_scene_graph()
                        
                        if is_foreground and self.remapped_idx[classes[idx]] not in self.thing_classes:
                            if self.remapped_idx[classes[idx]] in self.always_stuff:
                                # Illegal class (e.g. "grass" inside "zebra") -> do not draw
                                continue
                        
                        # Determine type
                        if not is_foreground:
                            is_foreground = self.remapped_idx[classes[idx]] in self.thing_classes
                        
                        candidate.append({
                            'mask': bool_mask,
                            'class': classes[idx],
                            'children': [],
                            'score': scores[idx],
                            'bbox': boxes[idx],
                            'idx': idx,
                            'original_class_name': original_class_name,
                            'is_foreground': is_foreground,
                        })
                        render_queue.append(candidate[-1])
                        
                for element in render_queue:
                    bool_mask = element['mask']
                    cl = element['class']
                    score = element['score']
                    idx = element['idx']
                    
                    if element['is_foreground']:
                        out_mask_fg[bool_mask] = self.remapped_idx[cl]
                    else:
                        assert self.remapped_idx[cl] <= len(self.stuff_classes), self.remapped_idx[cl]
                        out_mask_bg[bool_mask] = self.remapped_idx[cl]
                    out_mask[bool_mask] = self.remapped_idx[cl]
                    drawn_masks.append(bool_mask)
                    drawn_masks_indices.append(self.remapped_idx[cl])
                    drawn_masks_scores.append(score)
                    mask_ref[bool_mask] = len(drawn_masks)

                    # Prepare attributes
                    mask_attributes = []
                    if region_db is not None:
                        mask_attributes = set()
                        original_class_name = element['original_class_name']
                        if original_class_name in region_db:
                            for region in region_db[original_class_name]:
                                if 'attribute_synsets' in region:
                                    bb = region['x'], region['y'], region['w'], region['h']
                                    bb_adapted = element['bbox'].copy()
                                    bb_adapted[2:] -= bb_adapted[:2]

                                    overlap = iou_bb(bb_adapted, bb)
                                    if overlap > 0.5:
                                        mask_attributes.update(region['attribute_synsets'])
                                    #print('overlap {} {}'.format(original_class_name, overlap))

                        mask_attributes = list(mask_attributes)
                        mask_attributes.sort(key=lambda x: -self.attribute_counter[x])
                        if self.attribute_hook is not None:
                            self.attribute_hook(idx, original_class_name, mask_attributes)
                        for i in range(num_attr_masks):
                            if len(mask_attributes) > i and mask_attributes[i] in self.attr_map:
                                attr_masks[i][bool_mask] = self.attr_map[mask_attributes[i]]
                                if element['is_foreground']:
                                    attr_masks_fg[i][bool_mask] = self.attr_map[mask_attributes[i]]
                                else:
                                    attr_masks_bg[i][bool_mask] = self.attr_map[mask_attributes[i]]
                            else:
                                attr_masks[i][bool_mask] = 0
                                if element['is_foreground']:
                                    attr_masks_fg[i][bool_mask] = 0
                                else:
                                    attr_masks_bg[i][bool_mask] = 0

                    if debug:
                        tmp = mask_debug[bool_mask]
                        tmp.fill((idx, mask_attributes))
                        mask_debug[bool_mask] = tmp
                    #print(original_class_name, mask_attributes)

        if debug:
            return out_mask, out_mask_bg, out_mask_fg, mask_debug, attr_masks, attr_masks_bg, attr_masks_fg
        else:
            return out_mask, out_mask_bg, out_mask_fg, attr_masks, attr_masks_bg, attr_masks_fg

    def __getitem__(self, index):
        # input image (real images)
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        assert self.paths_match(label_path[:-4], image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        exif = image._getexif() if hasattr(image, '_getexif') else None
        if exif is not None:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(exif.items())
            if orientation in exif:
                if exif[orientation] == 3:
                    image=image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image=image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image=image.rotate(90, expand=True)
        
        image = image.convert('RGB')
        res = image.size
        
        # Label Image
        if self.debug_mode:
            label_mask, label_mask_bg, label_mask_fg, self.debug_mask, attr_masks, attr_masks_bg, attr_masks_fg = \
                self.make_mask(label_path, res, debug=True)
        else:
            label_mask, label_mask_bg, label_mask_fg, attr_masks, attr_masks_bg, attr_masks_fg = self.make_mask(label_path, res)
        
        
        label = Image.fromarray(label_mask.astype(np.int16))
        params = get_params(self.opt, label.size)

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)
        
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        
        label_tensor = transform_label(label)
        
        input_dict = {
            'image': image_tensor,
            'label': label_tensor,
            'path': image_path,
        }
        
        if self.opt.two_step_model:
            label_fg = Image.fromarray(label_mask_fg.astype(np.int16))
            label_bg = Image.fromarray(label_mask_bg.astype(np.int16))
            label_tensor_fg = transform_label(label_fg)
            label_tensor_bg = transform_label(label_bg)
            
            input_dict['label_bg'] = label_tensor_bg
            input_dict['label_fg'] = label_tensor_fg
        
        if self.opt.embed_attributes:
            attributes = []
            for mask in attr_masks:
                attributes.append(Image.fromarray(mask.astype(np.int16)))
            attr_tensors = [transform_label(x) for x in attributes]
            attr_tensor = torch.stack(attr_tensors, dim=0)
            input_dict['attributes'] = attr_tensor
                
            if self.opt.two_step_model:
                attributes_fg = []
                for mask in attr_masks_fg:
                    attributes_fg.append(Image.fromarray(mask.astype(np.int16)))
                attr_tensors_fg = [transform_label(x) for x in attributes_fg]
                attr_tensor_fg = torch.stack(attr_tensors_fg, dim=0)
                input_dict['attributes_fg'] = attr_tensor_fg

                attributes_bg = []
                for mask in attr_masks_bg:
                    attributes_bg.append(Image.fromarray(mask.astype(np.int16)))
                attr_tensors_bg = [transform_label(x) for x in attributes_bg]
                attr_tensor_bg = torch.stack(attr_tensors_bg, dim=0)
                input_dict['attributes_bg'] = attr_tensor_bg
                
            
        # Load captions
        if self.opt.embed_captions and self.opt.use_precomputed_captions:
            text_path = self.precomputed_captions_paths[index]
            assert self.paths_match(label_path[:-4], text_path, cast_to_int=True), \
                    "The label_path %s and image_path %s don't match." % \
                    (label_path, text_path)
                
            with torch.no_grad():
                captions = torch.load(text_path)
                max_len = 32 # We want a fixed memory layout to avoid sudden OOM errors
                if self.opt.isTrain:
                    caption_id = torch.randint(low=0, high=len(captions), size=(1,)).long().item()
                    token_embs = captions[caption_id][1]
                else:
                    if self.opt.concat_all_captions: # This is enabled by default for FID evaluation
                        caption_id = -1
                        capts = []
                        for capt in captions:
                            capts.append(capt[1])
                        token_embs = torch.cat(capts, dim=0)
                        max_len *= 5
                    else:
                        caption_id = self.manual_caption_id # Always use caption 0 when testing
                        token_embs = captions[caption_id][1]

                
                token_embs = token_embs[:min(max_len, token_embs.shape[0])]
                token_len = torch.LongTensor([token_embs.shape[0]])
                token_embs = F.pad(token_embs.t(), (0, max_len - token_len), mode='constant').t() # Zero pad
                
            input_dict['sentence_emb'] = token_embs
            input_dict['sentence_len'] = token_len

            if self.debug_mode:
                input_dict['caption_id'] = caption_id
                input_dict['sentence_text'] = captions[caption_id][0]
                input_dict['all_captions'] = captions
                    
        elif self.opt.embed_captions and not self.opt.use_precomputed_captions:
            image_id = int(os.path.basename(image_path).split('.')[0])
            captions = self.all_captions[image_id]
            
            with torch.no_grad():
                if self.opt.isTrain:
                    caption_id = torch.randint(low=0, high=len(captions), size=(1,)).long().item()
                    capts = [ captions[caption_id] ]
                else:
                    if self.opt.concat_all_captions: # This is enabled by default for FID evaluation
                        caption_id = -1
                        capts = captions[:5]
                        assert len(capts) == 5
                    else:
                        caption_id = self.manual_caption_id # Always use caption 0 when testing
                        capts = [ captions[caption_id] ]
                        
                tokenized_captions = []
                sentence_lengths = []
                for caption in capts:
                    tokens = self.tokenizer.encode(caption, add_special_tokens=True)
                    max_len = 32 # We want a fixed memory layout to avoid sudden OOM errors
                    tokens = torch.LongTensor(tokens[:min(max_len, len(tokens))])
                    token_len = torch.LongTensor([tokens.shape[0]])
                    tokens = F.pad(tokens, (0, max_len - token_len), mode='constant') # Zero pad
                    tokenized_captions.append(tokens)
                    sentence_lengths.append(token_len)
            
            input_dict['sentence_tokens'] = torch.stack(tokenized_captions, dim=0)
            input_dict['sentence_len'] = torch.stack(sentence_lengths, dim=0)
            
            if self.debug_mode:
                input_dict['caption_id'] = caption_id
                input_dict['sentence_text'] = captions[caption_id]
                input_dict['all_captions'] = captions
            

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
