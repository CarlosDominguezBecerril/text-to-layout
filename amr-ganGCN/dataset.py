import json
from torch.utils.data import Dataset
from collections import defaultdict
import torch
import numpy as np
import os

class CocoDataset(Dataset):
    
    def __init__(self, graphs_path, instan_path, normalize=True, vocab=None, image_size=(256, 256), 
    amr=True, uq_cap=False, max_objects=10, all_objects_valid=True, include_image=-1):
        
        # Paths of the file
        self.graphs_path = graphs_path
        self.instan_path = instan_path

        self.normalize = normalize

        # image_size
        self.image_size = image_size
        
        # dataset information
        self.amr = amr
        self.uq_cap = uq_cap
        self.max_objects = max_objects
        self.all_objects_valid = all_objects_valid
        self.include_image = include_image

        # Load all the image ids, filenames, sizes, objects and triples using the graph info
        graph_data = None
        with open(self.graphs_path, "r") as json_file:
            graph_data = json.load(json_file)
            
            # Generate all the basic information about each picture   
            self.image_ids = []
            self.image_id_to_filename = {}
            self.image_id_to_size = {}
            self.image_id_to_triples = {}
            self.image_id_to_object_list = {}
            self.vocab = vocab
            self.num_of_valid_object = {}
            self.image_id_to_caption = {}

            for image_id in graph_data.keys():
                # Only for unique caption 
                max_obj, max_obj_id, number_of_ones_id = 0, 0, 0
                for i in range(graph_data[image_id]['valid_captions']):
                    number_of_ones = 0
                    # Each image can have MORE than one caption therefore we create strings
                    # of type "00001-1" for the first caption "00001-2" for the second caption
                    # and so on
                    if self.amr:
                        if len(graph_data[image_id]['graphs'][i]['triples']) == 0:
                            continue
                    
                        all_ceros_k2 = True
                        for k1, k2 in graph_data[image_id]['graphs'][i]['objects']:
                            if k2 == 1:
                                all_ceros_k2 = False
                                number_of_ones += 1
                                
                        if all_ceros_k2:
                            continue

                    if self.uq_cap:
                        # Take the caption with the most objects
                        if len(graph_data[image_id]['graphs'][i]['objects']) > max_obj:
                            max_obj = len(graph_data[image_id]['graphs'][i]['objects'])
                            max_obj_id = i
                            number_of_ones_id = number_of_ones
                    else:
                        # The information in this loop is unique to each caption
                        image_id_c  = str(image_id) + "-" + str(i)
                        self.image_ids.append(image_id_c)
                        self.image_id_to_triples[image_id_c] = graph_data[image_id]['graphs'][i]['triples']
                        self.image_id_to_object_list[image_id_c] = graph_data[image_id]['graphs'][i]['objects']
                        self.num_of_valid_object[image_id_c] = number_of_ones if self.amr else len(graph_data[image_id]['graphs'][i]['objects'])
                        self.image_id_to_caption[image_id_c] = graph_data[image_id]['graphs'][i]['caption']
                        
                if self.uq_cap:
                    if len(graph_data[image_id]['graphs']) == 0:
                        continue
                    elif self.amr and number_of_ones_id == 0:
                        continue
                    elif not self.amr and max_obj == 0:
                        continue
                    
                    if self.amr and max_obj == 0:
                        print("ERROR")
                    self.image_ids.append(image_id)
                    self.image_id_to_triples[image_id] = graph_data[image_id]['graphs'][max_obj_id]['triples']
                    self.image_id_to_object_list[image_id] = graph_data[image_id]['graphs'][max_obj_id]['objects']
                    self.num_of_valid_object[image_id] = number_of_ones_id if self.amr else max_obj
                    self.image_id_to_caption[image_id] = graph_data[image_id]['graphs'][max_obj_id]['caption']

                # The information that is the same for all the captions
                self.image_id_to_filename[image_id] = graph_data[image_id]['image_filename']
                width, height = graph_data[image_id]['width'], graph_data[image_id]['height']
                self.image_id_to_size[image_id] = (width, height)

        vocab_remove = False     
        if self.vocab == None:
            vocab_remove = True
            # Special words '__image__', '__in_image__' as shown in the paper
            # Extra word __<UNK>__ for unknown tokens
            self.vocab = {
                'object_name_to_idx': {"__image__":0, "__<UNK>__": 1},
                'object_idx_to_name': {0:"__image__", 1: "__<UNK>__"},
                'pred_name_to_idx': {"__in_image__":0, "__<UNK>__": 1},
                'pred_idx_to_name': {0:"__in_image__", 1: "__<UNK>__"},
                'word2index': {"<pad>": 0, "<sos>":1, "<eos>": 2, "<unk>":3},
                'index2word': {0:"<pad>", 1:"<sos>", 2:"<eos>", 3:"<unk>"},
                "word2count": {"<pad>": 0, "<sos>":len(self.image_ids), "<eos>":len(self.image_ids), "<unk>":0},
                "index2wordCoco": {},
                "word2indexCoco": {},
                "index2wordint": {0:"<pad>", 1:"<sos>", 2:"<eos>", 3:"<unk>"}
            }

            
            # Objects
            seen_obj = set()
            p = 0
            
            for key, value in self.image_id_to_object_list.items():
                for obj in value:
                    if self.amr:
                        obj = obj[0]
                    if obj in seen_obj or obj.lower() in seen_obj:
                        continue
                    seen_obj.add(obj)
                    seen_obj.add(obj.lower())
                    # Original. +2 because we have __in_image__ and __<UNK>__
                    self.vocab['object_name_to_idx'][obj.lower()] = p+2
                    self.vocab['object_idx_to_name'][p+2] = obj.lower()
                    p += 1
                
            # Relations  
            seen_rel = set()    
            p = 0
            for key, value in self.image_id_to_triples.items():
                for rel in value:
                    if rel[1] in seen_rel or rel[1].lower() in seen_rel:
                        continue
                    seen_rel.add(rel[1])
                    seen_rel.add(rel[1].lower())
                    # Original. +2 because we have __in_image__ and __<UNK>__
                    self.vocab['pred_name_to_idx'][rel[1].lower()] = p+2
                    self.vocab['pred_idx_to_name'][p+2] = rel[1].lower()
                    p += 1
            
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
                    # word = self.vocab['index2wordCoco'][object_data['category_id']]
                    # self.vocab['word2count'][word] += 1

        # Delete the instances that has no coco objects
        total = 0
        for id in self.image_ids:
            new = id if self.uq_cap else id.split("-")[0]
            if new not in self.image_id_to_objects:
                self.image_ids.remove(id)
                total += 1
        
        print("Number of graphs removed from the list without gt objects {}".format(total))
        if vocab_remove:
            self.vocab['word2count']["<sos>"] -= total
            self.vocab['word2count']["<eos>"] -= total
        
        del self.num_of_valid_object, self.image_id_to_filename, 
                
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
        return self.image_ids[idx]
    
    def get_image_caption(self, idx):
        return self.image_id_to_caption[idx]

    def __getitem__(self, idx):
        """
            This function returns information about the idx-th caption in image_ids list
            Output:
            
            objects     -> Objects that the picture has
            triples     -> Triples showing the relationship between objects
            valid_objs  -> Objects that are going to be used after the GCN
            coco_boxes  -> Bounding boxes for each object
            coco_ids    -> coco id for each bounding box
        
        """
        
        # Load the information
        image_id = self.image_ids[idx]
        out_idx = idx
        img_id = image_id.split("-")[0]
        
        # Apply transformations to the pictures
        
        objs, valid_objs = [], []
        for word in self.image_id_to_object_list[image_id]:
            if self.amr:
                idx = 1 if word[0].lower() not in self.vocab['object_name_to_idx'] else self.vocab['object_name_to_idx'][word[0].lower()]
                if self.all_objects_valid:
                    valid_objs.append(1)
                else:
                    valid_objs.append(word[1])
            else:
                idx = 1 if word.lower() not in self.vocab['object_name_to_idx'] else self.vocab['object_name_to_idx'][word.lower()]
                valid_objs.append(1)
            objs.append(idx)
        
            
        # Add dummy __image__ object
        objs.append(self.vocab['object_name_to_idx']['__image__'])
        valid_objs.append(self.include_image)
        
        objs = torch.LongTensor(objs)

        triples = []
        # Add triples
        for triple in self.image_id_to_triples[image_id]:
            # Triples information to know with which object it matches
            s, o = triple[0][1], triple[2][1]
            p = 1 if triple[1].lower() not in self.vocab['pred_name_to_idx'] else self.vocab['pred_name_to_idx'][triple[1].lower()]
            triples.append([s, p, o])
        
        # Add __in_image__ triples to connect all the graph
        O = objs.size(0)
        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([i, in_image, O - 1])

        # Obtain the associated objects
        boxes_coco, ids_coco = self.get_coco_objects_tensor(out_idx)

        # Convert to long tensors
        triples = torch.LongTensor(triples)
        valid_objs = torch.LongTensor(valid_objs)
        out_idx = torch.LongTensor([out_idx])
        
        return objs, triples, valid_objs, boxes_coco, ids_coco, out_idx

def coco_collate_fn(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving object categories
    - boxes: FloatTensor of shape (O, 4)
    - triples: LongTensor of shape (T, 3) giving triples
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - triple_to_img: LongTensor of shape (T,) mapping triples to images
    """
    # all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
    all_objs, all_triples = [], []
    all_obj_to_img, all_triple_to_img = [], []
    all_valid_objects = []
    all_boxes_coco, all_ids_coco, all_coco_to_img = [], [], []
    all_idx = []
    obj_offset = 0
    # for i, (img, objs, boxes, triples, match) in enumerate(batch):
    for i, (objs, triples, valid_objects, boxes_coco, ids_coco, idx) in enumerate(batch):
        # all_imgs.append(img[None])
        if objs.dim() == 0 or triples.dim() == 0:
            continue
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes_coco.append(boxes_coco)
        all_ids_coco.append(ids_coco)
        all_idx.append(idx)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)
        all_valid_objects.append(valid_objects)
        all_coco_to_img.append(torch.LongTensor(boxes_coco.size(0)).fill_(i))
        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_objs = torch.cat(all_objs)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)
    all_valid_objects = torch.cat(all_valid_objects)
    all_boxes_coco = torch.cat(all_boxes_coco)
    all_ids_coco = torch.cat(all_ids_coco)
    all_coco_to_img = torch.cat(all_coco_to_img)
    all_idx = torch.cat(all_idx)
    out = (all_objs, all_triples,
             all_obj_to_img, all_triple_to_img, all_valid_objects, all_boxes_coco, all_ids_coco, all_coco_to_img, all_idx)
    return out

    

    

    