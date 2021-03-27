import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np

import random

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, vocab, is_training, max_len=12, teacher_learning=True):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.teacher_learning = teacher_learning
        self.output_l_size = len(vocab['index2word'])
        self.is_training = is_training
        self.max_len = max_len
        self.temperature = 0.4
        self.xy_distribution_size = self.decoder.xy_distribution_size

    def change_is_training(self, is_training):
        self.is_training = is_training
        self.decoder.is_training = is_training

    def convert_objects_to_list(self, obj_to_img, coco_boxes, coco_ids, coco_to_img):
        # Reorginize the target
        # Longest sequence (so we can add padding after)
        maximum = 3
        target_l_variables_list, target_x_variables_list, target_y_variables_list, target_w_variables_list, target_h_variables_list = [], [], [], [], []
        target_coco_to_img_list = []
        for i in range(obj_to_img.max()+1):
            object_idx = np.where(coco_to_img.cpu().numpy()==i)[0]
            target_l_variables_list.append(coco_ids[object_idx[0]: object_idx[-1]+1])
            target_x_variables_list.append(coco_boxes[:, 0][object_idx[0]: object_idx[-1]+1])
            target_y_variables_list.append(coco_boxes[:, 1][object_idx[0]: object_idx[-1]+1])
            target_w_variables_list.append(coco_boxes[:, 2][object_idx[0]: object_idx[-1]+1])
            target_h_variables_list.append(coco_boxes[:, 3][object_idx[0]: object_idx[-1]+1])
            target_coco_to_img_list.append(coco_to_img[object_idx[0]: object_idx[-1]+1])
            maximum = max(maximum, len(target_l_variables_list[-1]))

        # Add padding
        target_l_variables, target_x_variables, target_y_variables, target_w_variables, target_h_variables = [], [], [], [], []
        target_coco_to_img = []
        for i in range(len(target_l_variables_list)):
            s = len(target_l_variables_list[i])
            target_l, target_x, target_y, target_w, target_h, target_c_t_i = torch.zeros(maximum, dtype=torch.int64), torch.zeros(maximum), torch.zeros(maximum), torch.zeros(maximum), torch.zeros(maximum), torch.empty(maximum).fill_(i)
            target_l[:s], target_x[:s], target_y[:s], target_w[:s], target_h[:s], target_c_t_i[:s] = target_l_variables_list[i], target_x_variables_list[i], target_y_variables_list[i], target_w_variables_list[i], target_h_variables_list[i], target_coco_to_img_list[i]
            target_l_variables.append(target_l)
            target_x_variables.append(target_x)
            target_y_variables.append(target_y)
            target_w_variables.append(target_w)
            target_h_variables.append(target_h)
            target_coco_to_img.append(target_c_t_i)
        target_l_variables = torch.stack(target_l_variables)
        target_x_variables = torch.stack(target_x_variables)
        target_y_variables = torch.stack(target_y_variables)
        target_w_variables = torch.stack(target_w_variables)
        target_h_variables = torch.stack(target_h_variables)
        target_coco_to_img = torch.stack(target_coco_to_img)
        
        target_x_coordinates = torch.zeros((target_x_variables.shape[0], target_x_variables.shape[1]))
        target_y_coordinates = torch.zeros((target_y_variables.shape[0], target_y_variables.shape[1]))
        for i in range(target_x_variables.shape[1]):
            coordinates = self.convert_to_coordinates(self.convert_from_coordinates(torch.cat((target_x_variables[:, i].unsqueeze(1), target_y_variables[:, i].unsqueeze(1)), dim=1)).argmax(1).unsqueeze(1))

            target_x_coordinates[:, i] = coordinates[:, 0]
            target_y_coordinates[:, i] = coordinates[:, 1]

        target_x_variables = target_x_coordinates
        target_y_variables = target_y_coordinates
        
        if torch.cuda.is_available():
            target_l_variables, target_x_variables, target_y_variables, target_w_variables, target_h_variables, target_coco_to_img = target_l_variables.cuda(), target_x_variables.cuda(), target_y_variables.cuda(), target_w_variables.cuda(), target_h_variables.cuda(), target_coco_to_img.cuda()
        return target_l_variables, target_x_variables, target_y_variables, target_w_variables, target_h_variables, target_coco_to_img

    def forward(self, objs, triples, obj_to_img, triple_to_img, valid_objs, coco_boxes, coco_ids, coco_to_img):
        
        # Store the matches
        l_match, total = 0, 0

        # Obtain the targets
        target_l, target_x, target_y, target_w, target_h, target_coco_to_img = self.convert_objects_to_list(obj_to_img, coco_boxes, coco_ids, coco_to_img)

        # Encode the input
        encoder_output, encoder_hidden, include = self.encoder(objs, triples, obj_to_img, valid_objs)
        
        decoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])

        # context = decoder_hidden[0]

        batch_size = decoder_hidden[0].size(1)

        # Maximum target_length (12)
        if self.is_training:
            trg_len = target_l.size(1)
        else:
            trg_len = min(target_l.size(1), self.max_len)

        # tensor to store decoder outputs
        if self.is_training:
            outputs_class = torch.zeros(trg_len, target_l.size(0), self.decoder.output_size)
            outputs_bbox = torch.zeros(trg_len, target_l.size(0), 4)
            outputs_bbox_xy = torch.zeros(trg_len, target_l.size(0), self.xy_distribution_size ** 2)
            target_xy = torch.zeros(trg_len, target_l.size(0))
            # outputs_bbox_xy = torch.zeros(trg_len, batch_size, 2)
        else:
            outputs_class = torch.zeros(trg_len, batch_size, self.decoder.output_size)
            outputs_bbox = torch.zeros(trg_len, batch_size, 4)
            outputs_bbox_xy = torch.zeros(trg_len, batch_size, self.xy_distribution_size ** 2)
            target_xy = torch.zeros(trg_len, batch_size)
            # outputs_bbox_xy = torch.zeros(trg_len, batch_size, 2)
        
        if torch.cuda.is_available():
            outputs_class = outputs_class.cuda()
            outputs_bbox = outputs_bbox.cuda()
            outputs_bbox_xy = outputs_bbox_xy.cuda()
            target_xy = target_xy.cuda()

        # Obtain <sos> label
        if self.is_training:
            trg_l = target_l[:, 0] # <sos>
        else:
            trg_l = torch.ones(batch_size, dtype=torch.long)
            if torch.cuda.is_available():
                trg_l = trg_l.cuda()
        
        # Obtain bbox for each label (all zeros for <sos>)
        if self.is_training:
            trg_x, trg_y, trg_w, trg_h = target_x[:, 0], target_y[:, 0], target_w[:, 0], target_h[:, 0]
        else:
            trg_x = torch.zeros(batch_size, dtype=torch.float)
            trg_y = torch.zeros(batch_size, dtype=torch.float)
            trg_w = torch.zeros(batch_size, dtype=torch.float)
            trg_h = torch.zeros(batch_size, dtype=torch.float)
            if torch.cuda.is_available():
                trg_x = trg_x.cuda()
                trg_y = trg_y.cuda()
                trg_w = trg_w.cuda()
                trg_h = trg_h.cuda()

        teacher_force = random.random() < 0.5
        teacher_force = True
        if self.is_training and self.teacher_learning and teacher_force:
            next_l = torch.FloatTensor(target_l.size(0), self.output_l_size)
            if torch.cuda.is_available():
                next_l = next_l.cuda()
                
            # Prediction of the next class (softmax). Only used for training.
            next_l[next_l != 0] = 0
            for batch_index in range(target_l.size(0)):
                next_l[batch_index, int(target_l[batch_index, 1])] = 1
            
            next_xy = torch.cat((target_x[:, 1].unsqueeze(1), target_y[:, 1].unsqueeze(1)), dim=1)
        else:
            next_l = None
            next_xy = None

        # Column by column
        for di in range(1, trg_len):
            # Decoder
            class_prediction, xy_out, wh_out, decoder_hidden, xy_coordinates = self.decoder(trg_l, trg_x, trg_y, trg_w, trg_h, decoder_hidden, encoder_output, next_l=next_l, next_xy=next_xy)

            outputs_class[di] = class_prediction
            outputs_bbox[di, :, 2:] = wh_out

            # https://pytorch-nlp-tutorial-ny2018.readthedocs.io/en/latest/day2/sampling.html
            if xy_coordinates == None:
                xy_distance = xy_out.div(self.temperature).exp()
                xy_topi = torch.multinomial(xy_distance, 1)
                xy_coordinates = self.convert_to_coordinates(xy_topi)
            # xy_coordinates = self.convert_to_coordinates(xy_out.argmax(1).unsqueeze(1))

            outputs_bbox[di, :, :2] = xy_coordinates
            outputs_bbox_xy[di] = xy_out
            # outputs_bbox_xy[di] = next_xy

            """
            outputs_class[di] = next_l
            outputs_bbox[di, :, 0] = target_x[:, di]
            outputs_bbox[di, :, 1] = target_y[:, di]
            outputs_bbox[di, :, 2] = target_w[:, di]
            outputs_bbox[di, :, 3] = target_h[:, di]
            """

            target_xy[di] = self.convert_from_coordinates(torch.cat((target_x[:, di].unsqueeze(1), target_y[:, di].unsqueeze(1)), dim=1)).argmax(1)

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < 0.5
            teacher_force = True

            top1 = class_prediction.argmax(1)
            if self.is_training and self.teacher_learning and teacher_force:
                trg_l, trg_x, trg_y, trg_w, trg_h = target_l[:, di], target_x[:, di], target_y[:, di], target_w[:, di], target_h[:, di]
            else:
                trg_l = top1
                trg_x = xy_coordinates[:, 0]
                trg_y = xy_coordinates[:, 1]
                trg_w = wh_out[:, 0]
                trg_h = wh_out[:, 1]
            
            if self.is_training and self.teacher_learning and teacher_force:
                if di == trg_len-1:
                    next_l = None
                    next_xy = None
                else:
                    # Prediction of the next class (softmax). Only used for training
                    if next_l == None:
                        next_l = torch.FloatTensor(target_l.size(0), self.output_l_size)
                        if torch.cuda.is_available():
                            next_l = next_l.cuda()
                    next_l[next_l != 0] = 0
                    for batch_index in range(target_l.size(0)):
                        next_l[batch_index, int(target_l[batch_index, di+1])] = 1

                    next_xy = torch.cat((target_x[:, di+1].unsqueeze(1), target_y[:, di+1].unsqueeze(1)), dim=1)
            else:
                next_l = None
                next_xy = None

            if self.is_training:
                # Calculate some stats about the output (correct matching without taking into account the padding)
                target_tensor = target_l[:, di]
                non_padding = target_tensor.ne(self.vocab['word2index']['<pad>'])
                l_correct = top1.view(-1).eq(target_tensor).masked_select(non_padding).sum().item()
                l_match += l_correct
                total += non_padding.sum().item()

        final_output = {
            "output_class": outputs_class,
            "output_bbox": outputs_bbox,
            "outputs_bbox_xy": outputs_bbox_xy,
            "target_l": target_l,
            "target_x": target_x,
            "target_y": target_y,
            "target_w": target_w,
            "target_h": target_h,
            "target_xy": target_xy,
            "l_match": l_match,
            "total": total,
            "coco_to_img": target_coco_to_img
        }
        return final_output

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if h.shape[0] == 2:
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def convert_from_coordinates(self, input_coordinates):
        distribution = torch.zeros((input_coordinates.shape[0], self.xy_distribution_size, self.xy_distribution_size), device=input_coordinates.device, dtype=torch.int)
        # We make a matrix to make the operations easier
        # distribution = [batch_size, xy_distribution_size, xy_distribution_size]

        # Convert from 0, 1 to 0, xy_distribution_size
        input_coordinates[:, 0] = (input_coordinates[: ,0] * self.xy_distribution_size).clamp(0, self.xy_distribution_size-1)
        input_coordinates[:, 1] = (input_coordinates[: ,1] * self.xy_distribution_size).clamp(0, self.xy_distribution_size-1)

        input_coordinates = input_coordinates.long()
        for i in range(input_coordinates.shape[0]):
            distribution[i, input_coordinates[i, 1], input_coordinates[i , 0]] = 1

        return torch.flatten(distribution, 1)

    def convert_to_coordinates(self, input_coordinates):
        # To obtain y coordinate -> (input_coordinates[i] / number of sectors
        # ### Check if there is an easier way to calculate x
        # To obtain x coordinate -> ((input_coordinates[i] * number_of_sectors) % (number_of_sectors ** 2)) /  number_of_sectors
        # 0 | 1 | 2 | 3
        # - - - - - - -
        # 4 | 5 | 6 | 7
        # - - - - - - -
        # 8 | 9 | 10 | 11
        # - - - - - - -
        # 12 | 13 | 14 | 15
        # 
        # number_of_sectors = 4 # 4 rows and 4 columns
        # 
        # Ex.
        # input coordinate: 3
        # y = (3 / 4) = 0.75 = int(0.75) = 0 -> row 0
        # x = ((3 * 4) % 16) / 4 = ((12) % 16) / 4 = 12 / 4 = 3 -> col 3
        # 
        # input coordinate: 6
        # y = (6 / 4) = 1.5 = int(1.5) = 1 -> row 1 
        # x = ((6 * 4) % 16) / 4 = ((24) % 16) / 4 = 8 / 4 = 2 -> col 2
        # 
        # input coordinate: 12
        # y = (12 / 4) = 3.0 = int(3.0) = 3 -> row 3
        # x = ((12*4) % 16) / 4 = ((48) % 16) / 4 = 0 / 4 = 0 -> col 0
        number_of_sectors = self.xy_distribution_size

        # First obtain the coordinates of the matrix
        x, y = ((input_coordinates*number_of_sectors) % (number_of_sectors**2)).floor_divide(number_of_sectors), input_coordinates.floor_divide(number_of_sectors)

        # Obtain the [x,y] value in [0, 1] range
        x_value = x.true_divide(number_of_sectors)
        y_value = y.true_divide(number_of_sectors)

        return torch.cat((x_value, y_value), dim=1)
