## Revision in Continuous Space: Unsupervised Text Style Transfer without Adversarial Learning
We provide a TensorFlow implementation of the **Revision in Continuous Space: Unsupervised Text Style Transfer without Adversarial Learning**, AAAI 2020.

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
- **TextCNN.ipynb**: Pretrain a Text-CNN on the train set for predictor training.
- **TextBiLSTM.ipynb**: Pretrain a Text-BiLSTM on the whole dataset for evaluation
- **KenLM / Moses**: Pretrain a language model. **Text\_Style\_Transfer\_Pipeline.ipynb**: The pipeline (training, inference, and evaluation) for text sentiment transfer and text gender style transfer.
- **Multi\_Finegrained\_Control.ipynb**: The pipeline (training, and inference) for multiple fine-grained attributes control.
- **Eval\_Multi.ipynb**: The Evaluation of the multiple fine-grained attributes control.

