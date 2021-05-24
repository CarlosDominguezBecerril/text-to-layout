import json
from torch.utils.data import Dataset
from collections import defaultdict
import torch
import numpy as np
import os

from transformers import BertTokenizer

# from nltk.tokenize import RegexpTokenizer

class CocoDataset(Dataset):
    
    def __init__(self, dataset_path, instan_path, normalize=True, vocab=None, image_size=(256, 256), uq_cap=False, max_objects=10):
        
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
            self.seen = {}
            
            for annot in dataset_data['annotations']:
                # If we are using one caption take the first one that appears
                image_id = annot['image_id']
                if self.uq_cap and image_id in self.seen:
                    continue

                if image_id in self.seen:
                    image_id_c = str(image_id) + "-" + str(self.seen[image_id])
                    self.seen[image_id] += 1
                else:
                    image_id_c = str(image_id) + "-" + "0"
                    self.seen[image_id] = 1

                self.image_ids.append(image_id_c)
                self.image_id_to_caption[image_id_c] = annot['caption']

            # add information about the picture
            for image in dataset_data['images']:
                image_id, height, width, filename = image['id'], image['height'], image['width'], image['file_name']
                self.image_id_to_filename[str(image_id)] = filename
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
            new = id.split("-")[0]
            if new not in self.image_id_to_objects:
                self.image_ids.remove(id)
                total += 1
        
        print("Number of captions removed from the list without gt objects {}".format(total))
        if vocab_remove:
            self.vocab['word2count']["<sos>"] -= total
            self.vocab['word2count']["<eos>"] -= total

        
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

        boxes_coco, ids_coco = self.get_coco_objects_tensor(out_idx)
        out_idx = torch.LongTensor([out_idx])

        return caption, boxes_coco, ids_coco, out_idx


class Collator():

    def __init__(self):
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __call__(self, batch):

        # Captions
        captions = []

        # objects
        all_boxes_coco, all_ids_coco, all_coco_to_img = [], [], []
        all_idx = []

        for i, (caption, boxes_coco, ids_coco, idx) in enumerate(batch):
            # Captions
            captions.append(caption)

            # Objects
            all_coco_to_img.append(torch.LongTensor(boxes_coco.size(0)).fill_(i))
            all_boxes_coco.append(boxes_coco)
            all_ids_coco.append(ids_coco)
            all_idx.append(idx)

        encoding = self.tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
        all_inputs_ids = encoding['input_ids']
        all_attention_masks = encoding['attention_mask']

        all_boxes_coco = torch.cat(all_boxes_coco)
        all_ids_coco = torch.cat(all_ids_coco)
        all_coco_to_img = torch.cat(all_coco_to_img)
        all_idx = torch.cat(all_idx)

        out = (all_inputs_ids, all_attention_masks, all_boxes_coco, all_ids_coco, all_coco_to_img, all_idx)
        return out