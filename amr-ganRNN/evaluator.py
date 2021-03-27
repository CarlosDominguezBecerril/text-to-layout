from tqdm import tqdm

import torch
from torch.nn import functional as F

import json

import collections

import numpy as np

from loss import xy_distribution_loss

class Evaluator():
    def __init__(self, seq2seq, loss_class, loss_bbox_xy, loss_bbox_wh, vocab, gaussian_dict=None, validator_output_path="./", save_output=False, verbose=False, name="DEVELOPMENT"):
        self.seq2seq = seq2seq
        self.loss_class = loss_class
        self.loss_bbox_xy = loss_bbox_xy
        self.loss_bbox_wh = loss_bbox_wh
        self.vocab = vocab
        self.VERBOSE = verbose
        self.save_output = save_output
        self.validator_output_path = validator_output_path if validator_output_path[-1] != "/" else validator_output_path[:-1]
        self.gaussian_dict = gaussian_dict
        self.name = name

    def convert_index_to_word(self, lxywh):
        return [(self.vocab['index2wordCoco'][l[-1]]) for l in lxywh]
    
    def clean_output(self, l, x, y, w, h):
        output = []
        for i in range(len(l)):
            if l[i] in [0, 1, 2, 3]:
                break
            output.append([torch.clamp(x[i], 0, 1).item(), torch.clamp(y[i], 0, 1).item(), torch.clamp(w[i], 0, 1).item(), torch.clamp(h[i], 0, 1).item(), int(self.vocab['index2wordint'][int(l[i])])])
        return output
    
    def filter_redundant_labels(self, predicted):
        ls = np.array([i[-1] for i in predicted])
        # filter redundant labels
        counter = collections.Counter(ls)
        unique_labels, label_counts = list(counter.keys()), list(counter.values())
        kept_indices = []
        for label_index in range(len(unique_labels)):
            label = unique_labels[label_index]
            label_num = label_counts[label_index]
            # sample an upper-bound threshold for this label
            mu, sigma = self.gaussian_dict[label]
            threshold = max(int(np.random.normal(mu, sigma, 1)), 2)
            old_indices = np.where(ls == label)[0].tolist()
            new_indices = old_indices
            if threshold < len(old_indices):
                new_indices = old_indices[:threshold]
            kept_indices += new_indices
        kept_indices.sort()
        final_output = []
        for i in range(len(predicted)):
            if i in kept_indices:
                final_output.append(predicted[i])
        return final_output
        
    @torch.no_grad()
    def evaluate(self, dl, ds, epoch, output_folder):
        
        self.seq2seq.eval()
        self.seq2seq.change_is_training(False)
        self.seq2seq.teacher_learning = False

        epoch_loss, epoch_lloss, epoch_bloss_xy, epoch_bloss_wh, epoch_bloss_xy_MSE = 0, 0, 0, 0, 0
        
        bbox_and_ls_outputs = {}
        step = 0
        for batch in tqdm(dl):
            
            captions_padded, captions_length, coco_boxes, coco_ids, coco_to_img, all_idx = batch
            outputs = self.seq2seq(captions_padded, captions_length, coco_boxes, coco_ids, coco_to_img)
            
            # Class LOSS
            output_dim = outputs['output_class'].shape[-1]
            output_class_clean = outputs['output_class'][1:]
            outputs_class =  output_class_clean.view(-1, output_dim)
            # output_class = [trg len*batch size, output dim]
            
            target_l = outputs['target_l'][:, 1:]
            target_l_cleaned = torch.transpose(target_l, 0, 1).contiguous().view(-1)
            # target_l = [batch size*output dim]

            class_loss = self.loss_class(outputs_class, target_l_cleaned)

            # BBOX LOSS
            # cross entropy xy <start>
            output_xy_dim = outputs['outputs_bbox_xy'].shape[-1]
            output_xy_clean = outputs['outputs_bbox_xy'][1:]
            outputs_xy = output_xy_clean.view(-1, output_xy_dim)

            target_xy = outputs['target_xy'][1:].contiguous().view(-1).long()
            xy_prob_loss = self.loss_bbox_xy(outputs_xy, target_xy)
            # cross entropy xy <end>

            output_dim = outputs['output_bbox'].shape[-1]
            output_bbox_clean = outputs['output_bbox'][1:]
            outputs_bbox =  output_bbox_clean.view(-1, output_dim)

            target_x, target_y, target_w, target_h, target_coco_to_img = outputs['target_x'][:, 1:], outputs['target_y'][:, 1:], outputs['target_w'][:, 1:], outputs['target_h'][:, 1:], outputs['coco_to_img'][:, 1:]

            # Concatenate the [x, y, w, h] coordinates
            target_xywh = torch.zeros(outputs_bbox.shape)
            target_coco = torch.zeros(outputs_bbox.shape[0])

            if torch.cuda.is_available():
                target_xywh = target_xywh.cuda()
                target_coco = target_coco.cuda()
            trg_len = outputs['target_l'].shape[1]
            batch_size = outputs['target_l'].shape[0]
            for di in range(trg_len-1): # -1 because we remove the first object
                target_xywh[di*batch_size:di*batch_size+batch_size, 0] = target_x[:, di]
                target_xywh[di*batch_size:di*batch_size+batch_size, 1] = target_y[:, di]
                target_xywh[di*batch_size:di*batch_size+batch_size, 2] = target_w[:, di]
                target_xywh[di*batch_size:di*batch_size+batch_size, 3] = target_h[:, di]
                target_coco[di*batch_size:di*batch_size+batch_size] = target_coco_to_img[:, di]

            output_compare = torch.zeros(len(output_class_clean), outputs['target_l'].size(0))
            for di in range(len(output_class_clean)):
                # Step di
                output_compare[di] = output_class_clean[di].argmax(1)

            # After finding <pad>, <ukn>, <sos> or <eos> token convert the remaining numbers to 0
            for di in range(len(output_class_clean)):
                for dj in range(outputs['target_l'].size(0)):
                    if output_compare[di, dj] <= 3:
                        output_compare[di:, dj] = 0

            if torch.cuda.is_available():
                output_compare = output_compare.cuda()

            # mask padding and eos of target_xywh and output_bbox
            flatten = torch.flatten(output_compare)

            mask = (flatten != 0).float()

            if torch.cuda.is_available():
                mask = mask.cuda()
            """
            wh_loss, xy_loss = 0, 0
            bbox_loss = F.mse_loss(outputs_bbox, target_xywh, reduction="none").sum(1)
            bbox_loss = bbox_loss * mask # mask to remove padding
            mask_sum = torch.sum(mask)
            if int(mask_sum.item()) == 0:
                bbox_loss = torch.sum(bbox_loss)
            else:
                bbox_loss = torch.sum(bbox_loss) / mask_sum
            """
            wh_loss, xy_loss = self.loss_bbox_wh(outputs_bbox, target_xywh, mask=mask)
            wh_loss, xy_loss =  wh_loss * 10, xy_loss * 10
            
            # Total loss
            loss = class_loss + wh_loss + xy_prob_loss
            
            epoch_loss += loss.item()
            epoch_lloss += class_loss.item()
            epoch_bloss_wh += wh_loss.item()
            epoch_bloss_xy += xy_prob_loss.item()
            epoch_bloss_xy_MSE += xy_loss.item()
            
            if self.VERBOSE or self.save_output:                
                for idx in range(max(coco_to_img)+1):
                    image_id = ds.get_image_id(all_idx[idx])

                    cleaned_output_original = self.clean_output(target_l[idx], target_x[idx], target_y[idx], target_w[idx], target_h[idx])
                    cleaned_output_predicted = self.clean_output(output_compare[:, idx], output_bbox_clean[:, idx, 0], output_bbox_clean[:, idx, 1], output_bbox_clean[:, idx, 2], output_bbox_clean[:, idx, 3])
                    if self.gaussian_dict != None:
                        cleaned_output_predicted = self.filter_redundant_labels(cleaned_output_predicted)
                    
                    if self.VERBOSE:
                        print("Results")
                        print("Caption", ds.image_id_to_caption[image_id])
                        print("Triples", ds.image_id_to_triples[image_id])
                        print("Original",  self.convert_index_to_word(cleaned_output_original))
                        print("predicted2", self.convert_index_to_word(cleaned_output_predicted))
                        print(" ")
                    
                    if self.save_output:
                        if len(cleaned_output_predicted) > 0:
                            bbox_and_ls_outputs[image_id] = cleaned_output_predicted
            step += 1
        with open(output_folder + "/" + self.name + "losses" + str(epoch)+ ".txt", "w") as f:
            # We need to substract one to the dataloader because we delete the last batch.
            # Need to be fixed. Error in objgan code.
            info = "{} {} {} {}".format(str(epoch_lloss/(len(dl))), str(epoch_bloss_xy/(len(dl))), str(epoch_bloss_wh/(len(dl))), str(epoch_bloss_xy_MSE/(len(dl))))
            f.write(info)
        
            
        if self.save_output: 
            with open(self.validator_output_path + "/" + self.name + "epoch" + str(epoch)+".json", "w") as f:
                json.dump(bbox_and_ls_outputs, f)
        
        