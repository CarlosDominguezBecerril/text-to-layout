# text-to-layout
Text to layout using GCN or RNN

Download and save the data in ./GCN2LY or ./RNN2LY folder:

https://drive.google.com/file/d/1FQC2yEV6--yM2ejsOsILR7pOTeTknLmE/view?usp=sharing

The data contains:
- datasets: A folder containing the following datasets:
  - AMR2014: Training and Testing datasets with the captions unprocessed.
  - AMR2014train-dev-test: Training, development and testing datasets with the captions processed.
  - COCO-annotations: MSCOCO2014 training and testing annotations.
- text_encoder100.pth: Pretrained text encoder (DAMSM).
- captions.pickle: Vocabulary of the pretrained encoder.
