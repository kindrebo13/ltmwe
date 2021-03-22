# -*- coding: utf-8 -*-
"""
Builds a word embedding Modified Latent Tree Model (MLTM) for use in feature
extraction from text. All non-leaf nodes from the tree are considered latent
variables.

Text classification tests are used to evaluate the 
features, in comparison with a naive Bayes classifier. The standard dataset
here is AG news topic classification. 

Preprocessed, filtered data is required. Original data can be found at:
https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv

@author: Kevin Indrebo
"""

import numpy as np
import pandas as pd
from argparse import Namespace
import torch
import utils
from utils import BinaryMLTMVectorizer,split_data,TextDataset,classify, \
    MLPFactory,SimpleModelTrainer
from models import WordEmbedMLTM,NaiveBayesClassifier


args = Namespace(data_file='data/agn/agn_data.pkl.gz',
                 seed=1337,
                 train_split_pct=0.7,
                 val_split_pct=0.1,
                 test_split_pct=0.2,
                 embed_file="data/glove.6B.50d.txt",
                 min_word_count=3,
                 min_descendants=32,
                 batch_size=64,
                 epochs=50,
                 early_stopping_criteria=5,
                 hidden_sizes=[32,16],
                 activation='ReLU',
                 dropout=0.2,
                 lr=5e-4)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

#Read AG news DataFrame from preprocessed pickle file
text_df = pd.read_pickle(args.data_file)

#Split data into train/val/test sets after randomly permuting
split_pct = {'train':args.train_split_pct,'val':args.val_split_pct,
             'test':args.test_split_pct}
text_data = split_data(text_df,split_pct)

#Create base text datsaset for use in naive Bayes classification
print("Creating base text dataset")
txt_vec = lambda x : x
base_dataset = TextDataset(text_data,txt_vec,'Class',lazy=False,
                           x_convert_type=None)

#Run naive Bayes classifier and compute accuracy results
naive_bayes = NaiveBayesClassifier()

print("Running Naive Bayes Classifier...")
nb_pct,y,nb_pred = classify(base_dataset,naive_bayes)
nb_pct = np.round(100*nb_pct,2)#format as % 

#find the glove data at: https://nlp.stanford.edu/projects/glove/
print("Processing word embeddings...")
embed_df = utils.process_glove_data(args.embed_file)

#Generate modified latent tree modelf from word embeddings and prune
print("Building MLTM tree...")
mltm = WordEmbedMLTM(text_data['train'],embed_df,args.min_word_count,'Text')
if args.min_descendants > 1:
    mltm.prune_tree(args.min_descendants)

#Create vectorizer for latent tree model features
mltm_vectorizer = BinaryMLTMVectorizer(mltm)

#Generate feature dataset for use in MLP classification
print("Creating MLTM feature dataset")
mltm_dataset = TextDataset(text_data,mltm_vectorizer,'Class',lazy=False,
                           x_convert_type=np.float32)

#Generate MLP model for classification
mlp_factory = MLPFactory(mltm_dataset.get_num_features(),
                         mltm_dataset.get_class_labels(),
                         args.hidden_sizes,args.activation,args.dropout)
mlp_clf = mlp_factory.generate()

#Standard cross-entropy loss function for MLP training
loss_fn = torch.nn.CrossEntropyLoss()

#Train model on training dataset
trainer = SimpleModelTrainer(mltm_dataset,loss_fn,**vars(args))
mlp_clf,train_state = trainer.train_model(mlp_clf)

#Classify test data and record accuracy results
test_data = mltm_dataset.get_data('test',False)
X = test_data['x']
y_pred = mlp_clf(X).detach().numpy()
cls_idx = np.argmax(y_pred,axis=1)
y = test_data['y'].detach().numpy()
correct = cls_idx == y
mltm_pct = np.round(100*np.count_nonzero(correct)/len(correct),2)

#Report accuracy results for both models
print("Naive Bayes Accuracy: "+str(nb_pct)+"%, MLTM-MLP Accuracy: "+ \
      str(mltm_pct)+"%")



