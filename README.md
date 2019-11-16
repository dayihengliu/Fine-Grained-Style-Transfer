# Revision in Continuous Space: Unsupervised Text Style Transfer without Adversarial Learning

This repo contains the code and data of the following paper:
>**Revision in Continuous Space: Unsupervised Text Style Transfer without Adversarial Learning**, *Dayiheng Liu, Jie Fu, Yidan Zhang, Chris Pal, Jiancheng Lv*, AAAI20 [[arXiv]](https://arxiv.org/abs/1905.12304)

# Overview
<p align="center"><img width="90%" src="1.png"/></p> 

We explore a novel task setting for text style transfer, in which it is required to simultaneously manipulate multiple ﬁne-grained attributes. We propose to address it by revising the original sentences in a continuous space based on gradient-based optimization.


# Dataset
- The Yelp and Amazon of the text sentiment transfer task can be download at http://bit.ly/2LHMUsl or https://worksheets.codalab.org/worksheets/0xe3eb416773ed4883bb737662b31b4948/ 
- The Yelp of the text gender style transfer can be download at http://tts.speech.cs.cmu.edu/style_models/gender_classifier.tar

# Prerequisites
- Jupyter notebook 4.4.0
- Python 3.6
- Tensorflow 1.6.0+
- Numpy
- nltk 3.3
- kenlm 0.0.0
- Moses

# Usage
- `TextCNN.ipynb`: Pretrain a Text-CNN on the train set for predictor training.
- `TextBiLSTM.ipynb`: Pretrain a Text-BiLSTM on the whole dataset for evaluation
- `KenLM / Moses`: Pretrain a language model. 
- `Text_Style_Transfer_Pipeline.ipynb`: The pipeline (training, inference, and evaluation) for text sentiment transfer and text gender style transfer.
- `Multi_Finegrained_Control.ipynb`: The pipeline (training, and inference) for multiple fine-grained attributes control.
- `Eval_Multi.ipynb`: The Evaluation of the multiple fine-grained attributes control.

# Output Samples
To make it easier for other researchers to compare our methods, we release the outputs of our methods for YELP and AMAZON.

For each dataset, we provide three kinds of outputs (content-strengthen, content-style-balanced, and style-strengthen) of our method, which can be found in `outputs/`.


