import json
from torch.utils.data import Dataset
from collections import defaultdict
import torch
import numpy as np
import os

# from nltk.tokenize import RegexpTokenizer

class CocoDataset(Dataset):
    
    def __init__(self, dataset_path, instan_path, normalize=True, vocab=None, image_size=(256, 256), uq_cap=False, max_objects=10, idx2word=None, word2idx=None, cap_lang=None):
        
        # Paths of the file
        self.dataset_path = dataset_path
        self.instan_path = instan_path

        # Normalize input
        self.normalize = normalize

        # image_size
        self.image_size = image_size
        
        # dataset information
        self.uq_cap = uq_cap
        self.max_objects = max_objects

        # Load all the captions
        dataset_data = None
        with open(self.dataset_path, "r") as json_file:
            dataset_data = json.load(json_file)

            self.image_ids = []
            self.image_id_to_filename = {}
            self.image_id_to_size = {}
            self.image_id_to_caption = {}

            for image_id in dataset_data.keys():
                for i in range(dataset_data[image_id]['valid_captions']):
                    # If we are using one caption take the first one
                    if self.uq_cap:
                        image_id_c = str(image_id)
                    else:
                        # All the captions
                        # Each image can have MORE than one caption therefore we create strings
                        # of type "00001-1" for the first caption "00001-2" for the second caption
                        # and so on
                        image_id_c  = str(image_id) + "-" + str(i)
                    self.image_id_to_caption[image_id_c] = self.normalize_string(dataset_data[image_id]['graphs'][i]['caption'])
                    self.image_ids.append(image_id_c)

                    # if we are using one caption after adding one break
                    if self.uq_cap:
                        break

                # add information about the picture
                self.image_id_to_filename[str(image_id)] = dataset_data[image_id]['image_filename']
                width, height = dataset_data[image_id]['width'], dataset_data[image_id]['height']
                self.image_id_to_size[str(image_id)] = (width, height)

        vocab_remove = False 
        # Read coco categories
        if vocab == None:
            vocab_remove = True
            self.vocab = {
                'word2index': {"<pad>": 0, "<sos>":1, "<eos>": 2, "<unk>":3},
                'index2word': {0:"<pad>", 1:"<sos>", 2:"<eos>", 3:"<unk>"},
                "word2count": {"<pad>": 0, "<sos>":len(self.image_ids), "<eos>":len(self.image_ids), "<unk>":0},
                "index2wordCoco": {},
                "word2indexCoco": {},
                "index2wordint": {0:"<pad>", 1:"<sos>", 2:"<eos>", 3:"<unk>"}
            }

            with open(self.instan_path, 'r') as json_file:
                data = json.load(json_file)
                for categories in data['categories']:
                    # Objects that the COCO dataset use
                    category_id = categories['id']
                    category_name = categories['name']
                        
                    self.vocab['index2word'][len(self.vocab['index2word'])] = category_name
                    self.vocab['word2index'][category_name] = len(self.vocab['word2index'])
                    self.vocab['word2count'][category_name] = 0
                    self.vocab['index2wordCoco'][category_id] = category_name
                    self.vocab['word2indexCoco'][category_name] = category_id

            for key, value in self.vocab['index2word'].items():
                if key in [0, 1, 2, 3]:
                    continue
                self.vocab["index2wordint"][key] = self.vocab['word2indexCoco'][value]
        else:
            self.vocab = vocab

        # If we do not have a dictionary create one
        if cap_lang == None:
            self.cap_lang = Lang('caption')

            # copy the vocabulary if already given
            if idx2word != None and word2idx != None:
                self.cap_lang.copy_ext_dict(idx2word, word2idx)
                for image_id, caption in self.image_id_to_caption.items():
                    self.cap_lang.index_cap_words(caption)
            else:
                # Generate the vocabulary
                for image_id, caption in self.image_id_to_caption.items():
                    self.cap_lang.index_words(caption)
        else:
            self.cap_lang = cap_lang
            self.cap_lang.reset_word2count()


        # Load instances
        instances_data = None
        with open(self.instan_path, 'r') as json_file:
            instances_data = json.load(json_file) 
            # Add object data from instances
            self.image_id_to_objects = defaultdict(list)
            for object_data in instances_data['annotations']:
                image_id = object_data['image_id']
                if str(image_id) in self.image_id_to_filename:
                    self.image_id_to_objects[str(image_id)].append(object_data)

        # Delete the instances that has no coco objects
        total = 0
        for id in self.image_ids:
            new = id if self.uq_cap else id.split("-")[0]
            if new not in self.image_id_to_objects:
                self.image_ids.remove(id)
                total += 1
        
        print("Number of captions removed from the list without gt objects {}".format(total))
        if vocab_remove:
            self.vocab['word2count']["<sos>"] -= total
            self.vocab['word2count']["<eos>"] -= total

    # Lowercase, trim, and remove non-letter characters
    def normalize_string(self, s):
        # Now included in the dataset
        """
        cap = s.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(cap.lower())

        tokens_new = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0:
                tokens_new.append(t)
        return tokens_new
        """
        return s.split(" ")
        
    def get_coco_objects_tensor(self, idx):
        # Obtain the coco objects associated with idx
        image_id = self.image_ids[idx]
        img_id = image_id.split("-")[0]
        # Obtain original and target size
        WW, HH = self.image_id_to_size[img_id]
        H, W = self.image_size

        boxes, ids = [], []

        # add sos bbox and id
        boxes.append(torch.FloatTensor([0, 0, 0, 0]))
        ids.append(1)
        
        # add ground truth objects
        k = 0
        all_bbox = np.zeros((len(self.image_id_to_objects[img_id]), 5))

        for coco_obj in self.image_id_to_objects[img_id]:
            word = self.vocab['index2wordCoco'][coco_obj['category_id']]
            
            x, y, w, h = coco_obj['bbox']
            # Normalize [0, 1]
            x, y, w, h, x1, y1 = x / WW, y / HH, w / WW, h / HH, (x+w) / WW, (y+h) / HH
            x_mean, y_mean = (x + x1)*0.5, (y + y1)*0.5
            # Scale to our desired ouput size (not recommended)
            if not self.normalize:
                x, y, w, h, x1, y1, x_mean, y_mean = x * W, y * H, w * W, h * H, x1 * W, y1* H, x_mean * H, y_mean * W
                
            l = self.vocab['word2index'][word]
            boxes.append(torch.FloatTensor([x_mean, y_mean, w, h]))
            ids.append(int(l))
        
        # add EOS bbox and id
        boxes.append(torch.FloatTensor([0, 0, 0, 0]))  
        ids.append(2)
        
        boxes = torch.stack(boxes, dim=0)
        ids = torch.LongTensor(ids)

        # Reorder
        sizes = boxes[1:len(boxes)-1, 2] * boxes[1:len(boxes)-1, 3]
        sorted_indices = [0] + (torch.argsort(sizes) + 1).tolist()[::-1][:self.max_objects]  + [len(boxes)-1]
        boxes = boxes[sorted_indices, :]
        ids = ids[sorted_indices]
        return boxes, ids
    
    def indexes_from_sentence(self, lang, sentence):
        # Tokenize a sentece
        if "<sos>" in lang.word2index:
            seq = [lang.word2index["<sos>"]]
        else:
            seq = []
        if type(sentence) is list:
            words = sentence
        else:
            words = sentence.split(' ')
        for word in words:
            if word in lang.word2index:
                seq.append(lang.word2index[word])
        if "<eos>" in lang.word2index:
            seq.append(lang.word2index["<eos>"])
        return seq
        
    def __len__(self):
        # This function returns the number of 'captions' that the dataset has
        return len(self.image_ids)
    
    def get_image_id(self, idx):
        # This function returns the image_id at position idx
        return self.image_ids[idx]
    
    def get_image_caption(self, image_id):
        # this function return the caption given the image_id
        return self.image_id_to_caption[image_id]

    def __getitem__(self, idx):        
        # Load the information
        image_id = self.image_ids[idx]
        out_idx = idx
        img_id = image_id.split("-")[0]

        caption = self.image_id_to_caption[image_id]
        tokenized_caption = self.indexes_from_sentence(self.cap_lang, caption)
        boxes_coco, ids_coco = self.get_coco_objects_tensor(out_idx)
        out_idx = torch.LongTensor([out_idx])
        tokenized_caption = torch.LongTensor(tokenized_caption)
        return tokenized_caption, boxes_coco, ids_coco, out_idx

def pad_seq(seq, max_length, pad_token):
    # Add padding to the sequence with a maximum of max_length
    seq = torch.cat((seq, torch.LongTensor([pad_token for i in range(max_length - len(seq))], device=seq.device)))
    return seq

def coco_collate_fn(batch):

    # Captions information
    captions_length = [len(b[0]) for b in batch]
    max_cap_length = max(captions_length)
    captions_padded = [pad_seq(b[0], max_cap_length, 0) for b in batch]

    # objects
    all_boxes_coco, all_ids_coco, all_coco_to_img = [], [], []
    all_idx = []

    for i, (tokenized_caption, boxes_coco, ids_coco, idx) in enumerate(batch):
        all_coco_to_img.append(torch.LongTensor(boxes_coco.size(0)).fill_(i))
        all_boxes_coco.append(boxes_coco)
        all_ids_coco.append(ids_coco)
        all_idx.append(idx)
    
    captions_padded = torch.stack(captions_padded)
    captions_length = torch.LongTensor(captions_length)

    all_boxes_coco = torch.cat(all_boxes_coco)
    all_ids_coco = torch.cat(all_ids_coco)
    all_coco_to_img = torch.cat(all_coco_to_img)
    all_idx = torch.cat(all_idx)

    out = (captions_padded, captions_length, all_boxes_coco, all_ids_coco, all_coco_to_img, all_idx)
    return out

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.word2count = {"<pad>": 0, "<sos>": 0, "<eos>": 0, "<unk>": 0}
        self.index2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.n_words = 4 # Count default tokens

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word)

    def index_cap_words(self, sentence):
        for word in sentence:
            self.index_cap_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def index_cap_word(self, word):
        if word in self.word2index:
            self.word2count[word] += 1

    def reset_word2count(self):
        self.word2count = {key: 0 for key in self.word2index}

    def enrich_word2count(self):
        for word in self.word2index:
            if word not in self.word2count:
                self.word2count[word] = 0

    def reset_ext_word2count(self):
        self.word2count = {}
        for word in self.word2index:
            self.word2count[word] = 0

    def copy_dict(self, lang):
        self.word2index = lang.word2index
        self.word2count = lang.word2count
        self.index2word = lang.index2word
        self.n_words = lang.n_words
        self.reset_word2count()

    def copy_ext_dict(self, ixtoword, wordtoix):
        self.word2index = wordtoix
        self.index2word = ixtoword
        self.n_words = len(self.word2index)

        self.word2count = {}
        self.reset_ext_word2count()

    def set_word2index(self, word2index):
        self.word2index = word2index
        self.n_words = len(self.word2index)

    def set_index2word(self, index2word):
        self.index2word = index2word
        
    def increase_seos_count(self):
        self.word2count["<sos>"] += 1
        self.word2count["<eos>"] += 1