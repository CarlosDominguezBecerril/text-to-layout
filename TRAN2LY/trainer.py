from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from evaluator import Evaluator

from transformers import get_linear_schedule_with_warmup

class SupervisedTrainer():

    def __init__(self, seq2seq, vocab, epochs, print_every, loss_class, loss_bbox_xy, loss_bbox_wh, batch_size, hidden_size, max_lr, opt_func, steps_per_epoch, checkpoints_path="./checkpoints", gaussian_dict=None, validator_output_path="./evaluator_output", save_output=False):
        self.seq2seq = seq2seq

        self.vocab = vocab
        self.epochs = epochs
        self.print_every = print_every
        self.loss_class = loss_class
        self.loss_bbox_xy = loss_bbox_xy
        self.loss_bbox_wh = loss_bbox_wh
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.max_lr = max_lr
        self.opt_func = opt_func
        self.checkpoints_path = checkpoints_path if checkpoints_path[-1] != "/" else checkpoints_path[:-1]

        # Optimizer and scheduler
        # self.optimizer = self.opt_func(self.seq2seq.parameters(), self.max_lr, weight_decay=1e-4)
        # self.sched = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.max_lr, epochs=self.epochs, steps_per_epoch=steps_per_epoch)
        self.optimizer = self.opt_func(self.seq2seq.parameters(), lr=self.max_lr, eps=1e-8)
        self.sched = get_linear_schedule_with_warmup(self.optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=steps_per_epoch*self.epochs)

        # Generate the evaluator
        self.gaussian_dict = gaussian_dict
        self.validator_output_path = validator_output_path
        self.save_output = save_output
        self.evaluator = Evaluator(self.seq2seq, self.loss_class, self.loss_bbox_xy, self.loss_bbox_wh, self.vocab, gaussian_dict=self.gaussian_dict, validator_output_path=self.validator_output_path, save_output=self.save_output)
    
    def train_batch(self, batch_step, inputs_ids, attention_masks, coco_boxes, coco_ids, coco_to_img):
        """
        This function trains a batch
        """

        outputs = self.seq2seq(inputs_ids, attention_masks, coco_boxes, coco_ids, coco_to_img)
        
        # final_output keys = [
        #   "output_class", "output_bbox", "outputs_bbox_xy",
        #   "target_l","target_x", "target_y", "target_w",
        #   "target_h","target_xy", "l_match", "total",
        #   "coco_to_img"]
        
        # Obtain the accuracy of correctly predicted classes
        total, l_match = outputs['total'], outputs['l_match']
        if batch_step % self.print_every == 0:
            if total == 0:
                l_accuracy = float('nan')
            else:
                l_accuracy = l_match / total
            print('l_accuracy: {}. l_match: {}. total: {}'.format(l_accuracy, l_match, total))
        
        # Obtain the loss of the class
        output_dim = outputs['output_class'].shape[-1]
        output_class_clean = outputs['output_class'][1:]
        outputs_class = output_class_clean.view(-1, output_dim)
        # output_class = [trg len*batch size, output dim]
        
        target_l = torch.transpose(outputs['target_l'][:, 1:], 0, 1).contiguous().view(-1)
        # target_l = [batch size*output dim]
        
        class_loss = self.loss_class(outputs_class, target_l)
        
        # Obtain the loss of the the bounding box
        # cross entropy xy
        output_xy_dim = outputs['outputs_bbox_xy'].shape[-1]
        output_xy_clean = outputs['outputs_bbox_xy'][1:]
        outputs_xy = output_xy_clean.view(-1, output_xy_dim)

        target_xy = outputs['target_xy'][1:].contiguous().view(-1).long()
        xy_prob_loss = self.loss_bbox_xy(outputs_xy, target_xy)
        
        # MSE of xywh
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

        if torch.cuda.is_available():
            output_compare = output_compare.cuda()

        # mask for padding and <eos> of target_xywh and output_bbox
        mask = (target_l != 0).float() * (target_l != 2).float()

        flatten = torch.flatten(output_compare)

        # Convert the mask to cuda
        if torch.cuda.is_available():
            mask = mask.cuda()

        # Obtain the losses
        wh_loss, xy_loss = self.loss_bbox_wh(outputs_bbox, target_xywh, mask=mask)
        wh_loss, xy_loss =  wh_loss * 10, xy_loss * 10

        # Total loss
        loss = class_loss + wh_loss + xy_prob_loss

        self.seq2seq.zero_grad()
        
        loss.backward()
        
        nn.utils.clip_grad_value_(self.seq2seq.parameters(), 0.1)

        self.optimizer.step()
        
        self.sched.step()

        return class_loss.item(), wh_loss.item(), xy_prob_loss.item(), xy_loss.item()

    def train_epoches(self, train_loader, train_ds, val_loader, val_ds, start_epoch=0):
        
        steps_per_epoch = int(round(len(train_loader)))
        step = 0
        total_steps = steps_per_epoch * self.epochs
        
        for epoch in range(start_epoch, start_epoch + self.epochs):
            torch.cuda.empty_cache()
            # Set the model to training
            self.seq2seq.train()
            self.seq2seq.change_is_training(True)
            self.seq2seq.teacher_learning = True

            if self.seq2seq.freeze_encoder:
                self.seq2seq.encoder.eval()
            
            epoch_loss_total = 0  # Reset every epoch
            epoch_lloss_total = 0
            epoch_bloss_xy_total = 0
            epoch_bloss_wh_total = 0
            epoch_bloss_xy_MSE_total = 0

            print("Training epoch", epoch+1)
            # Train the model
            for batch in tqdm(train_loader):
                inputs_ids, attention_masks, coco_boxes, coco_ids, coco_to_img, all_idx = batch

                lloss, bloss_wh, bloss_xy, bloss_xy_MSE = self.train_batch(step, inputs_ids, attention_masks, coco_boxes, coco_ids, coco_to_img)
                
                epoch_loss_total += lloss + bloss_xy + bloss_wh
                epoch_lloss_total += lloss
                epoch_bloss_xy_total += bloss_xy
                epoch_bloss_wh_total += bloss_wh
                epoch_bloss_xy_MSE_total += bloss_xy_MSE
                
                if step % self.print_every == 0:
                    print('step: ', step)
                    
                    print_lloss_total = 0
                    print_bloss_xy_total = 0
                    print_bloss_wh_total = 0
                    print_loss_total = 0
                    print_bloss_xy_MSE_total = 0
                    print('{}/{} Progress: {}. {}: {}. {}:{}. {}:{}. {}:{}'.format(
                        step,
                        steps_per_epoch,
                        step / total_steps * 100,
                        'Cross Entropy Loss',
                        lloss,
                        'bbox XY loss',
                        bloss_xy,
                        'bbox WH loss',
                        bloss_wh,
                        "bbox XY MSE loss",
                        bloss_xy_MSE
                    ))
                step += 1
            
            # Save the epoch
            torch.save(self.seq2seq.state_dict(), self.checkpoints_path + '/amr-gan'+str(epoch)+'.pth')
            
            # Save the average losses
            with open(self.checkpoints_path + "/TRAININGlosses" + str(epoch)+ ".txt", "w") as f:
                info = "{} {} {} {}".format(str(epoch_lloss_total/(len(train_loader))), str(epoch_bloss_xy_total/(len(train_loader))), str(epoch_bloss_wh_total/(len(train_loader))), str(epoch_bloss_xy_MSE_total/(len(train_loader))))
                f.write(info)
                
            # Evaluate. 99% of the works but sometimes when sampling we get +inf and throws and exception. This is just to keep it training.
            try:
                self.evaluator.evaluate(val_loader, val_ds, epoch, self.checkpoints_path)
            except:
                pass