# Text To Layout
Text-to-layout using graph convolutional neural network (GCNN) or recurrent neural netowrk (RNN).

## Libraries
The code has been tested using Python 3.8.8 and pytorch 1.7.1.

Other libraries needed:
- tqdm (tested on 4.58.0)
- NumPy (tested on 1.19)
- nltk (tested on 3.5)

## Dataset
Download and save the data in ./GCN2LY or ./RNN2LY folder:

https://drive.google.com/file/d/1FQC2yEV6--yM2ejsOsILR7pOTeTknLmE/view?usp=sharing

The data contains:
- datasets: A folder containing the following datasets:
  - AMR2014: Training and Testing datasets with the captions unprocessed.
  - AMR2014train-dev-test: Training, development and testing datasets with the captions processed.
  - COCO-annotations: MSCOCO2014 training and testing annotations.
- text_encoder100.pth: Pretrained text encoder (DAMSM).
- captions.pickle: Vocabulary of the pretrained encoder.

## Training

To train the model you need to set up the following variables in *main.py* file:

- *IS_TRAINING*: True.
- *EPOCHS*: Number of epochs to train.
- *CHECKPOINTS_PATH*: The path to save the checkpoints (remember to create the folder before).

Additionally, you can set up the following variables to save the outputs of the development dataset.
- *SAVE_OUTPUT*: True.
- *VALIDATION_OUTPUT*: Path to store the output (remember to create the folder before).

For testing the training process is recommended to set the variable *UQ_CAP* in *main.py* to True.

## Testing
To test the model you need to set up the following variables in *main.py* file:

- *IS_TRAINING*: False.
- *EPOCH_VALIDATION*: The epoch number to validate.
- *VALIDATION_OUTPUT*: Path to store the output (remember to create the folder before).
- *SAVE_OUTPUT*: True.