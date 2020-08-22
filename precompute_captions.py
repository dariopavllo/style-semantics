import torch
import itertools
import json
from transformers import BertModel, BertTokenizer
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print('Warning: tqdm not found. Install it to see progress bar.')
    def tqdm(x): return x

print('Loading BERT...')
pretrained_weights = 'bert-base-uncased'
model = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True)
if torch.cuda.is_available():
    model = model.cuda()
    
model.eval()
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

def export_sentence_embeddings(input_data, directory, level):
    print(f'Saving to {directory}')
    with torch.no_grad():
        for im_id, captions in tqdm(input_data.items()):
            embeddings = []
            for text in captions:
                tokenized = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).cuda()
                _, _, all_hidden_states = model(tokenized)
                emb = all_hidden_states[level][0].cpu()
                embeddings.append((text, emb))
            
            torch.save(embeddings, directory + '/' + str(im_id) + '.bin')

def load_captions(filename):
    print('Loading captions from', filename)
    with open(filename) as f:
        captions = json.load(f)

    all_captions = {}
    for caption in captions['annotations']:
        image_id = caption['image_id']
        text = caption['caption'].strip(' ,.;!:')
        if image_id not in all_captions:
            all_captions[image_id] = []
        all_captions[image_id].append(text)
    print('Loaded {} captions'.format(len(all_captions)))
    return all_captions


bert_level = -2 # Second-to-last layer
output_directory_prefix = 'datasets/coco/precomputed_captions_level' + str(abs(bert_level))

all_captions_val = load_captions('datasets/coco/annotations/captions_val2017.json')
output_dir_val = output_directory_prefix + '_val2017'
Path(output_dir_val).mkdir(parents=True, exist_ok=True)
export_sentence_embeddings(all_captions_val, output_dir_val, bert_level)

all_captions_train = load_captions('datasets/coco/annotations/captions_train2017.json')
output_dir_train = output_directory_prefix + '_train2017'
Path(output_dir_train).mkdir(parents=True, exist_ok=True)
export_sentence_embeddings(all_captions_train, output_dir_train, bert_level)