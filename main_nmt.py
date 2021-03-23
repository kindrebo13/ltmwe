# -*- coding: utf-8 -*-
"""
Builds a word embedding Modified Latent Tree Model (MLTM) for use in feature
extraction from text. All non-leaf nodes from the tree are considered latent
variables. These features are injected into a GRU encoder-decoder
with attention neural machine translation (NMT). The script runs comparison
tests with/without the latent variables on a small subset of English->French
sentence translation data.

@author: Kevin Indrebo
"""

import numpy as np
import os
from argparse import Namespace
import torch
import utils
from utils import BinaryMLTMVectorizer, NMTVectorizerWithMLTM,NMTModelWithMLTM, \
    NMTModelTrainer,NMTDatasetWithMLTM
from PyTorchNLPBook_utils import NMTModel
from models import WordEmbedMLTM


args = Namespace(data_dir="data/nmt",
                 data_file="fra.txt",
                 vectorizer_file="data/nmt/vectorizer.json",
                 model_state_file="models/nmt/model.pth",
                 save_dir="models/nmt",
                 embed_file="data/glove.6B.50d.txt",
                 min_word_count=0,
                 min_descendants=0,
                 cuda=False,
                 device="cpu",
                 seed=1337,
                 learning_rate=5e-4,
                 batch_size=32,
                 num_epochs=50,
                 early_stopping_criteria=5,              
                 source_embedding_size=64, 
                 target_embedding_size=64,
                 encoding_size=48,
                 mltm_dropout=0.2)

np.random.seed(args.seed)
torch.manual_seed(args.seed)


#Filter sentences with specific start phrases like he is, she is, etc.
nmt_lines = utils.filter_nmt_file(os.path.join(args.data_dir,args.data_file))
#Create nmt DataFrame from text
text_df = utils.create_nmt_data(nmt_lines)

#find the glove data at: https://nlp.stanford.edu/projects/glove/
embed_df = utils.process_glove_data(args.embed_file)

train_data = text_df[text_df['split']=='train']
#Generate modified latent tree modelf from word embeddings and prune
mltm = WordEmbedMLTM(train_data,embed_df,args.min_word_count,'source_language')
if args.min_descendants > 1:
    mltm.prune_tree(args.min_descendants)

#Create vectorizer for latent tree model features
mltm_vectorizer = BinaryMLTMVectorizer(mltm)
mltm_length = len(mltm_vectorizer)
print("MLTM has "+str(mltm_length)+" latent variables")

#Create general vectorizer for NMTModel
vectorizer = NMTVectorizerWithMLTM.from_dataframe(text_df)
vectorizer.set_mltm_vectorizer(mltm_vectorizer)

#Create Dataset used for neural network training
dataset = NMTDatasetWithMLTM(text_df,vectorizer)

#NMT model used for baseline comparison
base_model = NMTModel(source_vocab_size=len(vectorizer.source_vocab), 
                 source_embedding_size=args.source_embedding_size, 
                 target_vocab_size=len(vectorizer.target_vocab),
                 target_embedding_size=args.target_embedding_size, 
                 encoding_size=args.encoding_size,
                 target_bos_index=vectorizer.target_vocab.begin_seq_index)

base_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
print("Base NMTModel has "+str(base_params)+" learnable parameters")

#Train baseline model and compute test set accuracy
base_trainer = NMTModelTrainer(base_model,dataset,args)
base_model = base_trainer.train()
base_acc = base_trainer.test()

#Modified NMTModel with latent tree model variables injected into context vector
mltm_model = NMTModelWithMLTM(source_vocab_size=len(vectorizer.source_vocab), 
                 source_embedding_size=args.source_embedding_size, 
                 target_vocab_size=len(vectorizer.target_vocab),
                 target_embedding_size=args.target_embedding_size, 
                 encoding_size=args.encoding_size,
                 target_bos_index=vectorizer.target_vocab.begin_seq_index,
                 mltm_length=mltm_length,mltm_dropout=args.mltm_dropout)

mltm_params = sum(p.numel() for p in mltm_model.parameters() if p.requires_grad)
print("NMTModelWithMLTM has "+str(mltm_params)+" learnable parameters")

#Train modified model and compute test set accuracy
mltm_trainer = NMTModelTrainer(mltm_model,dataset,args)
mltm_model = mltm_trainer.train()
mltm_acc = mltm_trainer.test()

#Report accuracy results for both models
print("Base Acc: "+str(base_acc)+", MLTM Acc: "+str(mltm_acc))





