from data.sparse_dataset import SparseDataset
from data.image_folder import make_dataset
import os.path

class VgSparseDataset(SparseDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--augmented_training_set', action='store_true',
                                help='add training images from the COCO unlabeled set')
        return super(VgSparseDataset, VgSparseDataset).modify_commandline_options(parser, is_train)
    
    def get_paths(self, opt):
        if opt.isTrain:
            if opt.augmented_training_set:
                filelist_path = 'datasets/vg/vg_train_augmented_image_list.txt'
            else:
                filelist_path = 'datasets/vg/vg_train_image_list.txt'
        else:
            filelist_path = 'datasets/vg/vg_val_image_list.txt'
            
        # Do not match images between VG and COCO, as we are not using COCO
        self.coco_match_attributes = False
        
        image_paths = []
        detections_paths = []
        
        with open(filelist_path, 'r') as f:
            filelist = f.readlines()
            filelist = [x.rstrip() for x in filelist]
        
        for f in filelist:
            if f.startswith('unlabeled'):
                image_paths.append(os.path.join('datasets/coco', f))
                detections_paths.append('datasets/coco/detections_' + f + '.npz')
            else:
                image_paths.append(os.path.join('datasets/vg', f))
                detections_paths.append('datasets/vg/detections_' + f + '.npz')
                
        precomputed_captions_paths = None # Unsupported on VG
        assert not opt.embed_captions, 'Unsupported on VG (Visual Genome does not contain captions)'
        
        return detections_paths, image_paths, precomputed_captions_paths
                                              