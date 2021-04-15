from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from model.encoderGCN import Sg2ImModel
from model.decoderRNN import DecoderRNN
from model.seq2seq import Seq2Seq
from evaluator import Evaluator
from loss import bbox_loss

from dataset import CocoDataset, coco_collate_fn
from gpu import DeviceDataLoader, get_default_device, to_device
from trainer import SupervisedTrainer

import math

import numpy as np

from tqdm import tqdm

import collections

import os

# Dataloader
BATCH_SIZE = 32
NUM_WORKERS = 4
SHUFFLE = True
PIN_MEMORY = True

# Dataset hyperparameters
IMAGE_SIZE = (256, 256)
INCLUDE_IMAGE = -1 # Include __image__ node to the lstm. Values: 1 to include __image__ otherwise any number.
UQ_CAP = False # Use one caption or all the captions. Values: False -> All the captions. True -> One caption
ALL_OBJECTS_VALID = True # Include all the objects nodes to the lstm. Values: True -> include all. False -> Only the selected ones.
HIDDEN_SIZE = 256 
MAX_OBJECTS = 10 # Maximum number of objects to use from the dataset
NORMALIZE_INPUT = True # Normalize the pictures to range [0, 1].
BIDIRECTIONAL = True # Use a bidirectional encoder
USE_ATTENTION = True # use attention in the decoder
XY_DISTRIBUTION_SIZE = 32 # Size of grid use in the picture to approximate the bounding boxes.

# Training
EPOCHS = 3 # Number of epochs to train
PRINT_EVERY = 50 # Print information about the model every n steps
IS_TRAINING = False # Set the model to training or validation. Values: True -> Training mode. False -> Validation mode
CHECKPOINTS_PATH = "./checkpoints/1" # Path to save the epochs and average losses

# Validation
CALCULATE_GAUSS_DICT = True # Gauss dictionary with means and std for the objects in the dataset Values: True -> calculates and saves the gaussian dict. False -> Uses the file located at GAUSS_DICT_PATH  
GAUSS_DICT_PATH = "./data/gaussian_dict_full.npy" # Path to the gauss dict
SAVE_OUTPUT = True # Whether to save or not the output (bbox and class for each picture) when validating. Values: True -> the output is saved. False -> The output is not saved
VALIDATION_OUTPUT = "./evaluator_output/1" # Path to save the output (bbox and class for each picture)
EPOCH_VALIDATION = 28 # Number of the epoch to validate

# Paths to the training, development and validation dataset
GRAPHS_PATH_TRAIN = "./data/datasets/AMR2014train-dev-test/GraphTrain.json"
INSTAN_PATH_TRAIN = "./data/datasets/COCO/annotations/instances_train2014.json"

GRAPHS_PATH_DEV = "./data/datasets/AMR2014train-dev-test/GraphDev.json"
INSTAN_PATH_DEV = "./data/datasets/COCO/annotations/instances_train2014.json"

GRAPHS_PATH_VAL = "./data/datasets/AMR2014train-dev-test/GraphTest.json"
INSTAN_PATH_VAL = "./data/datasets/COCO/annotations/instances_val2014.json"

def generate_dataset(graph_path_train, instan_path_train, graph_path_test, instan_path_test, 
                    normalize_input=True, uq_cap=False, include_image=False, max_objects=10,
                    shuffle=True, num_workers=4, pin_memory=True, batch_size=16):
    """
    Function to generate the dataset and dataloaders
    """

    # Create the dataset
    print("Loading training dataset")
    train_ds = CocoDataset(graph_path_train, instan_path_train, normalize=normalize_input, uq_cap=uq_cap, include_image=include_image, max_objects=max_objects)

    if IS_TRAINING:
        print("Counting valid objects...")
        for i in tqdm(range(len(train_ds))):
            for j in train_ds.get_coco_objects_tensor(i)[1]:
                if j == 1 or j == 2:
                    continue
                train_ds.vocab['word2count'][train_ds.vocab['index2word'][j.item()]] += 1
                
    print("Loading validation dataset")
    val_ds = CocoDataset(graph_path_test, instan_path_test, normalize=normalize_input, vocab=train_ds.vocab, uq_cap=uq_cap, include_image=include_image, max_objects=max_objects)

    print("Train length:", len(train_ds))
    print("Validation length:", len(val_ds))

    # Generate data loaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=coco_collate_fn)
    val_dl = DataLoader(val_ds, batch_size, num_workers=num_workers, pin_memory=pin_memory, collate_fn=coco_collate_fn)

    # Send dataset to GPU if available
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    return train_dl, val_dl, train_ds.vocab, train_ds, val_ds

def generate_encoder(vocab, bidirectional=True, hidden_size=128, image_size=(256, 256)):
    """
    Function to generate the encoder
    """
    encoder = Sg2ImModel(vocab, image_size=image_size, embedding_dim=64, gconv_dim=hidden_size, bidirectional=bidirectional)
    return encoder

def generate_decoder(vocab, is_training, use_attention=False, bidirectional=True, hidden_size=128, xy_distribution_size=16):
    """
    Function to generate the decoder
    """
    decoder = DecoderRNN(vocab, hidden_size, is_training, use_attention=use_attention, bidirectional=bidirectional, xy_distribution_size=xy_distribution_size)
    return decoder

def generate_losses(vocab, train_ds):
    """
    Function to generate the losses
    """

    # Class loss
    total_sum = sum(vocab['word2count'].values())
    weight = torch.zeros(len(vocab['word2index']))
    for word in vocab['word2index']:    
        index = vocab['word2index'][word]
        weight[index] = (1 - (vocab['word2count'][word]/total_sum))
    
    weight[0], weight[3] = 0, 0
    
    lloss = nn.CrossEntropyLoss(weight, ignore_index=0)
    
    # bbox loss
    bloss_xy = nn.CrossEntropyLoss()
    bloss_wh = bbox_loss

    # send losses to GPU if available
    device = get_default_device()
    lloss, bloss_xy, bloss_wh = to_device(lloss, device), to_device(bloss_xy, device), bloss_wh

    return lloss, bloss_xy, bloss_wh

def calculate_gaussian_dict(train_ds):
    """
    Function to calculate the gaussian dictionary.
    """
    print("Getting class stats")
    sta_dict, gaussian_dict = {}, {}
    for i in tqdm(range(len(train_ds))):
        labels = train_ds.get_coco_objects_tensor(i)[1].tolist()[1:-1] # Remove first and last object <sos> and <eos>
        counter = collections.Counter(labels)
        unique_labels, label_counts = list(counter.keys()), list(counter.values())
        for label_index in range(len(unique_labels)):
            label = train_ds.vocab['index2word'][unique_labels[label_index]]
            label = train_ds.vocab['word2indexCoco'][label]
            count = label_counts[label_index]
            if label not in sta_dict:
                sta_dict[label] = []
                sta_dict[label].append(count)
            else:
                sta_dict[label].append(count)
    for label in sta_dict:
        tmp_mean = np.mean(np.array(sta_dict[label]))
        tmp_std = np.std(np.array(sta_dict[label]))
        gaussian_dict[label] = (tmp_mean, tmp_std)
    np.save(GAUSS_DICT_PATH, gaussian_dict)

if __name__ == "__main__":

    # Generate the dataset
    if IS_TRAINING:
        valg = GRAPHS_PATH_DEV
        vali = INSTAN_PATH_DEV
    else:
        valg = GRAPHS_PATH_VAL
        vali = INSTAN_PATH_VAL
    train_dl, val_dl, vocab, train_ds, val_ds = generate_dataset(GRAPHS_PATH_TRAIN, INSTAN_PATH_TRAIN, valg, vali, uq_cap=UQ_CAP, batch_size=BATCH_SIZE, max_objects=MAX_OBJECTS)
    
    # Generate the seq2seq model
    encoder, decoder = generate_encoder(vocab, hidden_size=HIDDEN_SIZE), generate_decoder(vocab, IS_TRAINING, xy_distribution_size=XY_DISTRIBUTION_SIZE, use_attention=USE_ATTENTION, hidden_size=HIDDEN_SIZE)
    
    # +2 objects because we need to include the <sos> and <eos>
    seq2seq = Seq2Seq(encoder, decoder, vocab, IS_TRAINING, max_len=MAX_OBJECTS+2)
    
    # Move the model to GPU if available
    device = get_default_device()
    print("USING DEVICE", device)

    seq2seq = to_device(seq2seq, device)
    
    # Generate the losses
    lloss, bloss_xy, bloss_wh = generate_losses(vocab, train_ds)

    # Calculate gaussian dict and open the file
    if CALCULATE_GAUSS_DICT:
        calculate_gaussian_dict(train_ds)
    gaussian_dict = np.load(GAUSS_DICT_PATH, allow_pickle=True).item()

    # Train or validate
    if IS_TRAINING:
        train = SupervisedTrainer(seq2seq, vocab, EPOCHS, PRINT_EVERY, lloss, bloss_xy, bloss_wh, BATCH_SIZE, HIDDEN_SIZE, 1e-3, torch.optim.Adam, len(train_dl), checkpoints_path=CHECKPOINTS_PATH, gaussian_dict=gaussian_dict, validator_output_path=VALIDATION_OUTPUT, save_output=SAVE_OUTPUT)        
        train.train_epoches(train_dl, train_ds, val_dl, val_ds)
    else:
        # Epoch to validate
        epoch = EPOCH_VALIDATION
        seq2seq.load_state_dict(torch.load(CHECKPOINTS_PATH + "/amr-gan" + str(epoch) + ".pth"))
        evaluator = Evaluator(seq2seq, lloss, bloss_xy, bloss_wh, vocab, gaussian_dict=gaussian_dict, validator_output_path=VALIDATION_OUTPUT, save_output=True, verbose=False, name="TESTING")
        evaluator.evaluate(val_dl, val_ds, epoch, CHECKPOINTS_PATH)
