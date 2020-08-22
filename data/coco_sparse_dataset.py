from data.sparse_dataset import SparseDataset
from data.image_folder import make_dataset
import json

class CocoSparseDataset(SparseDataset):
    
    def get_paths(self, opt):
        if opt.isTrain:
            image_dir = 'datasets/coco/train2017'
            detection_dir = 'datasets/coco/detections_train2017'
            captions_path = 'datasets/coco/annotations/captions_train2017.json'
            precomputed_captions_dir = 'datasets/coco/precomputed_captions_level{}_train2017'.format(abs(opt.bert_embedding_level))
        else:
            image_dir = 'datasets/coco/val2017'
            detection_dir = 'datasets/coco/detections_val2017'
            captions_path = 'datasets/coco/annotations/captions_val2017.json'
            precomputed_captions_dir = 'datasets/coco/precomputed_captions_level{}_val2017'.format(abs(opt.bert_embedding_level))
            
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)
        detection_paths = make_dataset(detection_dir, recursive=False, read_cache=True)
        
        assert len(detection_paths) == len(image_paths), \
            "The #images in %s and %s do not match. Is there something wrong? {} {}".format(
               len(detection_paths), len(image_paths))
        
        if opt.embed_captions and opt.use_precomputed_captions:
            precomputed_captions_paths = make_dataset(precomputed_captions_dir, recursive=False, read_cache=True)
            assert len(detection_paths) == len(precomputed_captions_paths), \
                "The #text in %s and %s do not match. Is there something wrong? {} {}".format(
                len(detection_paths), len(precomputed_captions_paths))
            
        elif opt.embed_captions and not opt.use_precomputed_captions:
            precomputed_captions_paths = None
            
            # Load captions JSON
            print('Loading captions json....')
            with open(captions_path) as f:
                captions_json = json.load(f)
                
                all_captions = {}
                for caption in captions_json['annotations']:
                    image_id = caption['image_id']
                    text = caption['caption'].strip(' ,.;!:')
                    if image_id not in all_captions:
                        all_captions[image_id] = []
                    all_captions[image_id].append(text)
                self.all_captions = all_captions
            
        else:
            precomputed_captions_paths = None
            
        if opt.embed_attributes:
            assert not opt.embed_captions, 'Conditioning on both captions and attributes is unsupported'
            self.coco_match_attributes = True # Attributes have to be obtained by matching IDs to Visual Genome
        else:
            self.coco_match_attributes = False
            
        return detection_paths, image_paths, precomputed_captions_paths
            