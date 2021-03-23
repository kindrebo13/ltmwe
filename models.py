# -*- coding: utf-8 -*-
"""
Models for performing NLP tasks. Includes a Latent Tree Model class applied
to word embeddings, neural machine translation classes extending classes
from PyTorchNLPBook_utils.py, and a basic Naive Bayes Classifier.

@author: Kevin Indrebo
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from nltk import word_tokenize
from torch.nn import functional as F
import torch
import torch.nn as nn
from collections import Counter
from PyTorchNLPBook_utils import NMTEncoder,verbose_attention


class WordEmbedMLTM(object):
    """
    Represents a tree of words as leaves and latent variables as internal
    nodes. Performs AgglomerativeClustering on the input text data, then
    uses the linkage matrix to construct the tree.
    """

    def __init__(self,text_data,embed_df,min_word_count=0,text_col='Text'):
        """
        Parameters
        ----------
        text_data : pd.DataFrame
            DataFrame with a 'Text' columns.
        embed_file : str
            Filename for word embedding pickle file.
        min_word_count : int
            Minimum number of occurs for a word to be included
        text_col : str
            Name of the column within the data DF containing training text

        Returns
        -------
        None.
        """
        
        self.text_data = text_data
        self.build_tree(text_data,embed_df,min_word_count,text_col)

    def build_tree(self,text_data,embed_df,min_word_count=0,text_col='Text'):
        """
        Builds the tree using a linkage matrix from an agglomerative clustering,
        where the word embeddings are the features.
        
        Parameters
        ----------
        text_data : pd.DataFrame
            DataFrame with a 'Text' columns.
        embed_file : str
            Filename for word embedding pickle file.
        text_col : str
            Name of the column within the data DF containing training text
            
        Returns
        -------
        None.
        """
        
        #Filter embeddings by set of words contained within training data
        all_words_ = self.get_all_words(text_data,min_word_count,text_col)
        mask = embed_df.index.isin(all_words_)
        print("Filtering "+str(len(embed_df))+" words down to "+str(np.count_nonzero(mask)))
        embed_df = embed_df[mask]
        
        #Perform agglomerative clustering on embeddings, generating a full tree
        model = self.cluster(embed_df)
        linkage = model.children_
        words = embed_df.index.values
        n_samples = len(words)
        wc_mask = self.word_counts.index.isin(words)
        self.word_counts = self.word_counts[wc_mask]
        
        self.root = 2*n_samples-2
        self.n_leaves = n_samples
        self.words = words.copy()
        self.leaves = np.arange(n_samples,dtype=np.int32)
        self.non_leaves = np.arange(n_samples,2*n_samples-1,dtype=np.int32)
        self.all_nodes = np.append(self.leaves,self.non_leaves)
        self.word_idx_map = pd.Series(range(n_samples),index=self.words,dtype=int)
        
        #Build a map of node indexes to parent,child indexes
        self.parents = dict()
        self.children = dict()
        for i,pair in enumerate(linkage):
            parent = i + n_samples
            left_child = pair[0]
            right_child = pair[1]
            
            self.parents[left_child] = parent
            self.parents[right_child] = parent
            self.children[parent] = np.array([left_child,right_child],dtype=np.int32)

        self.precompute()

    def precompute(self):
        """
        Stores a number of maps allowing convenient, fast access to data such
        as list of ancestors by leaf nodes, and leaf_descendants by non-leaf
        node.

        Returns
        -------
        None.

        """
        
        n_samples = self.n_leaves
        #From parent map, build a full list of ancestors,descendants for every node
        self.ancestors = dict()
        self.descendants = dict()
        self.leaf_descendants = dict()
        for node,parent in self.parents.items():
            parent_list = []
            while parent is not None:#loop exits when we get to the root
                #add the parent to the list of ancestors for this node
                parent_list.append(parent)
                #get the list of leaf descendants for parent and add this node
                desc = self.descendants.get(parent,[])
                assert(node not in desc)
                desc.append(node)
                self.descendants[parent] = desc
                #if node is a leaf, also add it as leaf_descendant to parent
                if node < n_samples:
                    desc = self.leaf_descendants.get(parent,[])
                    assert(node not in desc)
                    desc.append(node)
                    self.leaf_descendants[parent] = desc
                #climb up the tree, getting parent of parent, if not root
                parent = self.parents.get(parent,None)
            #add list of ancestors to map for this node
            self.ancestors[node] = parent_list

        #Build map of descendant/leaf_descendant counts for non_leaf nodes
        self.n_descendants = {idx:len(self.descendants[idx]) for idx in \
                              self.non_leaves}
        self.n_leaf_descendants = {idx:len(self.leaf_descendants[idx]) for \
                                   idx in self.non_leaves}
        
        #Build 0-indexed map of depths for all nodes (root=0)
        self.depths = {idx:len(anc) for idx,anc in self.ancestors.items()}
        self.depths[self.non_leaves[-1]] = 0
    
    def get_all_words(self,data,min_word_count=0,text_col='Text'):
        """
        Creates a set of all words within the data.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame of training data.
        text_col : str
            Name of the column within the data DF containing training text
            
        Returns
        -------
        None.
        """
        
        word_counter = Counter()
            
        text = data[text_col]
        all_words_ = set()
        for text_i in text:
            words = word_tokenize(text_i.lower())#all lower case
            all_words_.update(words)
            
            for word in words:
                word_counter[word] += 1
        
        all_words_ = pd.Series(sorted(list(all_words_)))
        self.word_counts = pd.Series(word_counter)
        if min_word_count > 0:
            mask = self.word_counts > min_word_count
            qual_words = self.word_counts.index[mask]
            all_mask = all_words_.isin(qual_words).values
            all_words_ = all_words_[all_mask]
        
        return all_words_
    
    def cluster(self,embed_df):
        """
        Clusters the word embeddings using AgglomerativeClustering.

        Parameters
        ----------
        embed_df : pd.DataFrame
            DataFrame of word embeddings.

        Returns
        -------
        model : AgglomerativeClustering object.
            A fitted AgglomerativeClustering object.
        """
        
        print("Performing agglomerative clustering...")
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        data = embed_df.values
        model = model.fit(data)
        
        return model

    def prune_tree(self,min_leaf_descendants):
        """
        Prunes the tree by conglomerating leaves. Every leaf node will have to
        have at least min_leaf_descendants-1 siblings.
        Parameters
        ----------
        min_leaf_descendants : int
            Min size of leaf children for a parent node.

        Returns
        -------
        None.
        """
        
        #identify non-leaf nodes that have enough leaf descendants
        n_desc_series = pd.Series(self.n_leaf_descendants)
        min_mask = n_desc_series >= min_leaf_descendants
        n_desc_series = n_desc_series[min_mask.values]
        qual_nodes = n_desc_series.index
        
        #disqualify a non-leaf node if it has all qualified children
        def disqual(node):
            children_ = pd.Series(self.children[node])
            return np.all(children_.isin(qual_nodes))
        
        disqual_mask = qual_nodes.map(lambda x : disqual(x)).values.astype(bool)
        qual_nodes = qual_nodes[~disqual_mask]

        for node in qual_nodes:
            #set all leaf descendants as this nodes children
            desc = self.leaf_descendants[node]
            self.children[node] = np.array(desc,dtype=np.int32)
            assert(len(desc)==self.n_leaf_descendants[node])
            #set this node as parent for all leaf descendants
            for child in desc:
                self.parents[child] = node
            #remove non-leaf descendants from keys of children
            all_desc = self.descendants[node]
            for idx in all_desc:
                if idx in self.non_leaves:
                    self.children.pop(idx,None)
        
        self.non_leaves = np.array(sorted(list(self.children.keys())),dtype=np.int32)
        self.all_nodes = np.append(self.leaves,self.non_leaves).astype(np.int32)
        #recompute all stats using updated parent,child maps
        self.precompute()

    def __contains__(self,word):
        """
        Checks to see if word is one of the leaves of the tree.

        Parameters
        ----------
        word : str
            Word.

        Returns
        -------
        bool
            True if word is a leaf literal.

        """
        
        return word.lower() in self.words
    
    def word_ancestors(self,word):
        """
        Returns the list of ancestor nodes to the given word.

        Parameters
        ----------
        word : str
            Word.

        Returns
        -------
        list
            List of ancestor nodes to leaf node corresponding to word.
        """
        
        idx = self.word_idx_map[word.lower()]
        return self.ancestors[idx]


class NaiveBayesClassifier(object):
    """
    Simple classifier based on word probabilities. The probability of each 
    class given a word is computed as p(word|class)*p(class). The value 
    p(word) is ignored as its the same for each class, and doesn't affect the 
    max probability class. All probabilities are represented in log form.
    """
    
    def __init__(self,prob_floor=1e-5):
        """
        Parameters
        ----------
        prob_floor : float, optional
            Minimum probability for p(c|w) from training data. The default 
            is 1e-5.

        Returns
        -------
        None.
        """
        
        self.prob_floor = prob_floor
    
    def fit(self,text,classes):
        """
        Learns the prior and conditional probabilities for each class.

        Parameters
        ----------
        text : pd.Series
            Training strings.
        classes : pd.Series
            Training class labels.

        Returns
        -------
        None.
        """
        
        self.class_domain = sorted(list(set(classes)))
        
        words = set()
        for i,text_i in enumerate(text):
            words_i = word_tokenize(text_i)
            words.update(words_i)
        
        words = pd.Series(sorted(list(words)))
        
        self.priors = pd.Series(0,index=self.class_domain)
        self.word_counts = pd.DataFrame(0,index=words,columns=self.class_domain)
        
        for clazz in self.class_domain:
            class_mask = classes == clazz
            self.priors[clazz] = np.log2(np.count_nonzero(class_mask) / len(classes))
            sentences = text[class_mask]
            for sentence in sentences:
                words_i = word_tokenize(sentence)
                word_mask = words.isin(words_i).values
                self.word_counts.loc[word_mask,clazz] += 1
        
        self.conditionals = self.word_counts.copy()
        word_count_sums = self.word_counts.sum(axis=1)
        for clazz in self.class_domain:
            self.conditionals[clazz] /= word_count_sums
        self.conditionals = np.log2(np.maximum(self.prob_floor,self.conditionals))

    def predict(self,text):
        """
        Given a string or iterable of strings, generates a classification for
        each input element.

        Parameters
        ----------
        text : str/Iterable
            Strings to classify.

        Returns
        -------
        list/element
            Predicted class for example(s).
        """
        
        if np.isscalar(text):
            return self._classify(text)
        else:
            pred = []
            for txt in text:
                pred.append(self.classify(txt))
            return np.array(pred)
            
    def classify(self,text):
        """
        Predicts class for single element of text.

        Parameters
        ----------
        text : str
            String to classify.

        Returns
        -------
        object
            The predicted class.
        """
        
        log_probs = self.compute_log_probs(text)
        return log_probs.idxmax()

    def compute_log_probs(self,text):
        """
        Computes the log-probabilities for each class given a single
        element of text.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        log_probs : ndarray(float)
            log-probabilities for each class.

        """
        words = pd.Series(word_tokenize(text))
        mask = words.isin(self.conditionals.index.values).values
        words = words[mask].values
        log_probs = self.conditionals.loc[words,:].sum(axis=0)
        log_probs += self.priors
        return log_probs


class NMTModelWithMLTM(nn.Module):
    """
    Modification of NMTModel which injects a MLTM feature vector into
    the context vector of the NMTDecoder module.
    """
    
    def __init__(self, source_vocab_size, source_embedding_size, 
                 target_vocab_size, target_embedding_size, encoding_size, 
                 target_bos_index, mltm_length, mltm_dropout=None):
        """
        Parameters
        ----------
        source_vocab_size : int
            Number of unique words in source language.
        source_embedding_size : int
            Size of the source embedding vectors.
        target_vocab_size : int
            Number of unique words in target language.
        target_embedding_size : int
            Size of the target embedding vectors.
        encoding_size : int
            The size of the encoder RNN.
        target_bos_index : int
            Index for BEGIN-OF-SEQUENCE token.
        mltm_length : int
            Length of input MLTM feature vector.
        mltm_dropout : float, optional
            Dropout rate for output MLTM layer. The default is None.

        Returns
        -------
        None.
        """
        
        super().__init__()
        self.encoder = NMTEncoder(num_embeddings=source_vocab_size, 
                                  embedding_size=source_embedding_size,
                                  rnn_hidden_size=encoding_size)
        decoding_size = encoding_size * 2
        self.decoder = NMTDecoderWithMLTM(num_embeddings=target_vocab_size, 
                                  embedding_size=target_embedding_size, 
                                  rnn_hidden_size=decoding_size,
                                  bos_index=target_bos_index,
                                  mltm_length=mltm_length,
                                  mltm_dropout=mltm_dropout)
        
    def forward(self, x_source, x_mltm, x_source_lengths, target_sequence):
        """
        Forward pass of the model.

        Parameters
        ----------
        x_source : torch.Tensor
             The source text data tensor. x_source.shape should be 
             (batch, vectorizer.max_source_length)
        x_mltm : torch.Tensor
            The last hidden state in the NMTEncoder.
        x_source_lengths : torch.Tensor
            The length of the sequences in x_source.
        target_sequence : torch.Tensor
            The target text data tensor.

        Returns
        -------
        decoded_states : torch.Tensor
            Prediction vectors at each output step.
        """

        encoder_state, final_hidden_states = self.encoder(x_source, x_source_lengths)
        decoded_states = self.decoder(encoder_state=encoder_state, 
                                      initial_hidden_state=final_hidden_states, 
                                      target_sequence=target_sequence,
                                      x_mltm=x_mltm)
        return decoded_states


class NMTDecoderWithMLTM(nn.Module):
    """
    Modification of NMTDecoder which injects a MLTM feature vector into
    the context vector.
    """
    
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, bos_index,
                 mltm_length, mltm_dropout=None):
        """
        Parameters
        ----------
        num_embeddings : int
            Number of embeddings is also the number of unique words in target 
            vocabulary.
        embedding_size : int
            The embedding vector size.
        rnn_hidden_size : int
            Size of the hidden rnn state.
        bos_index : int
            Begin-of-sequence index.
        mltm_length : int
            Length of input MLTM feature vector.
        mltm_dropout : float, optional
            Dropout rate for MLTM output layer. The default is None.

        Returns
        -------
        None.
        """
        
        super().__init__()
        self.mltm_mlp = nn.Linear(mltm_length,rnn_hidden_size)
        if mltm_dropout is not None:
            self.mltm_dropout = nn.Dropout(mltm_dropout)
        else:
            self.mltm_dropout = None
        self._rnn_hidden_size = rnn_hidden_size
        self.target_embedding = nn.Embedding(num_embeddings=num_embeddings, 
                                             embedding_dim=embedding_size, 
                                             padding_idx=0)
        self.gru_cell = nn.GRUCell(embedding_size + rnn_hidden_size*2, 
                                   rnn_hidden_size)
        self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.classifier = nn.Linear(rnn_hidden_size * 3, num_embeddings)
        self.bos_index = bos_index
    
    def _init_indices(self, batch_size):
        """ return the BEGIN-OF-SEQUENCE index vector """
        return torch.ones(batch_size, dtype=torch.int64) * self.bos_index
    
    def _init_context_vectors(self, batch_size):
        """ return a zeros vector for initializing the context """
        return torch.zeros(batch_size, self._rnn_hidden_size)
        
    def forward(self, encoder_state, x_mltm, initial_hidden_state, target_sequence):
        """
        Forward pass of the model.

        Parameters
        ----------
        encoder_state : torch.Tensor
            The output of the NMTEncoder.
        x_mltm : torch.Tensor
            The last hidden state in the NMTEncoder.
        initial_hidden_state : torch.Tensor
            The target text data tensor.
        target_sequence : torch.Tensor
            Prediction vectors at each output step.

        Returns
        -------
        output_vectors : torch.Tensor
            Output prediction  vector.
        """

        #apply a linear transformation to project MLTM features to rnn_hidden_size
        y_mltm = self.mltm_mlp(x_mltm)
        if self.mltm_dropout is not None:
            y_mltm = self.mltm_dropout(y_mltm)
        
        # We are making an assumption there: The batch is on first
        # The input is (Batch, Seq)
        # We want to iterate over sequence so we permute it to (S, B)
        target_sequence = target_sequence.permute(1, 0)
        output_sequence_size = target_sequence.size(0)

        # use the provided encoder hidden state as the initial hidden state
        h_t = self.hidden_map(initial_hidden_state)

        batch_size = encoder_state.size(0)
        # initialize context vectors to zeros
        context_vectors = self._init_context_vectors(batch_size)
        #add the projected latent tree variables to the context vector
        context_vectors = torch.cat([context_vectors,y_mltm],dim=1)
        # initialize first y_t word as BOS
        y_t_index = self._init_indices(batch_size)
        
        h_t = h_t.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)

        output_vectors = []
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()
        
        for i in range(output_sequence_size):
            y_t_index = target_sequence[i]
                
            # Step 1: Embed word and concat with previous context
            y_input_vector = self.target_embedding(y_t_index)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)
            
            # Step 2: Make a GRU step, getting a new hidden vector
            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().detach().numpy())
            
            # Step 3: Use the current hidden to attend to the encoder state
            context_vectors, p_attn, _ = verbose_attention(encoder_state_vectors=encoder_state, 
                                                           query_vector=h_t)
            
            #add the projected latent tree variables to the context vector
            context_vectors = torch.cat([context_vectors,y_mltm],dim=1)
            # auxillary: cache the attention probabilities for visualization
            self._cached_p_attn.append(p_attn.cpu().detach().numpy())
            
            # Step 4: Use the current hidden and context vectors to make a prediction to the next word
            prediction_vector = torch.cat((context_vectors, h_t), dim=1)
            score_for_y_t_index = self.classifier(F.dropout(prediction_vector, 0.3))
            
            # auxillary: collect the prediction scores
            output_vectors.append(score_for_y_t_index)
            
        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)
        
        return output_vectors




