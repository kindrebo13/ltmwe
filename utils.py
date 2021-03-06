# -*- coding: utf-8 -*-
"""
Utility classes and functions for manipulation of data, generating and
training models, and running classification/translation tests.

@author: Kevin Indrebo
"""

import numpy as np
import pandas as pd
import copy
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
from nltk import word_tokenize
from tqdm.notebook import tqdm as tqdm_notebook
from PyTorchNLPBook_utils import NMTVectorizer,make_train_state, \
    update_train_state,generate_nmt_batches,sequence_loss,compute_accuracy, \
    NMTDataset,NMTSampler
from models import NMTModelWithMLTM


class TextDataset(Dataset):
    """
    A PyTorch compatible Dataset which stores a DataFrame containing features 
    derived from text, class labels, and (optionally) the original text strings. Splits the data
    into train, val (optional), and test sets.
    """
    
    def __init__(self,text_data,vectorizer,target_col,lazy=False,
                 x_convert_type=np.float32):
        """
        Parameters
        ----------
        text_data : dict
            Map of split names to data partitions.
        vectorizer : object
            Object which contains a vectorize method.
        target_col : str
            Name of the column with class labels.
        lazy : bool, optional
            Waits to vectorize until data is accessed. The default is False.
        x_convert_type : type, optional
            Type for converting data features. The default is np.float32.

        Returns
        -------
        None.
        """
        
        self.text_data = text_data
        self.vectorizer = vectorizer
        self.target_col = target_col
        self.x_convert_type = x_convert_type
        self.split_ = 'train'
        
        
        if lazy:
            self.processed_data = {key:None for key in self.text_data.keys()}
        else:
            self.process_data()
        
        self.set_split('train')
        
        class_labels = text_data['train'][target_col].unique()
        class_labels.sort()
        self.n_classes = len(class_labels)
        self.class_labels = pd.Series(range(self.n_classes),index=class_labels)

    def process_data(self):
        """
        Converts text data to feature data with the instance's vectorizer.

        Returns
        -------
        None.
        """
        
        self.processed_data = dict()
        for split,text_data_ in self.text_data.items():
            y = text_data_[self.target_col].values
            print("Vectorizing for split: "+split)
            x = np.array([self.vectorizer(x_) for x_ in text_data_['Text']])
                
            self.processed_data[split] = {'x':x,'y':y}
            
        self.set_split(self.split_)

    def set_split(self,split='train'):
        """
        Sets the current active partition.

        Parameters
        ----------
        split : str, optional
            Name of partition. The default is 'train'.

        Returns
        -------
        None.
        """
        
        self._target_data = self.processed_data[split]
        self.split_ = split

    def check_Data(self):
        """
        If data has not been processed, calls process_data.

        Returns
        -------
        None.
        """
        
        if self._target_data is None:
            self.processData()

    def __len__(self):
        self.check_Data()
        return len(self._target_data['y'])
    
    def __getitem__(self,idx):
        self.check_Data()
        y = self._target_data['y'][idx]
        y = self.class_labels[y]#convert label to index
        x = self._target_data['x'][idx]
        if self.x_convert_type is not None:
            x = x.astype(self.x_convert_type)
        return {'x':x,'y':y}
    
    def get_num_batches(self,batch_size):
        """
        Determines the number of batches.

        Parameters
        ----------
        batch_size : int
            size of batchese.

        Returns
        -------
        int
            Number of batches.
        """
        
        return len(self) // batch_size

    def get_num_features(self):
        """
        Returns the number of features in the processed data.

        Returns
        -------
        int
            Feature size.
        """
        
        return len(self[0]['x'])

    def get_class_labels(self):
        """
        Returns a list of the class labels.

        Returns
        -------
        list
            List of class labels..
        """
        
        y = self.get_data()['y']
        if type(y) == torch.Tensor:
            return y.unique().numpy()
        else:
            return sorted(list(set(y)))
    
    def lookup_class_idx(self,label):
        """
        Returns the index corresponding to the given class label.

        Parameters
        ----------
        label : str
            Class label.

        Returns
        -------
        int
            Class label index.
        """
        
        return self.class_labels[label]

    def get_data(self,split=None,numpy=True):
        """
        Returns ndarrays or Tensors of all data in the current split.

        Parameters
        ----------
        numpy : bool, optional
            If True, returns numpy ndarrays, else Tensors. The default is True.

        Returns
        -------
        dict
            A map of x/y to input,target data.
        """
        
        if split is not None:
            split_ = self.split_
            self.set_split(split)
        
        dataloader = DataLoader(self,batch_size=len(self),shuffle=False,
                                drop_last=False)
        
        for i,data_item in enumerate(dataloader):
            assert(i==0)
            x = data_item['x']
            y = data_item['y']
            if numpy:
                if type(x) == torch.Tensor:
                    x = x.detach().numpy()
                else:
                    x = np.array(x)
                if type(y) == torch.Tensor:
                    y = y.detach().numpy()
                else:
                    y = np.array(y)
        
        if split is not None:
            self.split_ = split_
        
        return {'x':x,'y':y}
        
    def get_n_folds(self,split=None,N=5,numpy=True,perm=None):
        """
        Partitions the full data into a list of ndarrays/Tensors.

        Parameters
        ----------
        split : str, optional
            The train/test split. The default is None.
        N : int, optional
            The number of partitions. The default is None.
        numpy : bool, optional
            Convert Tensors to numpy ndarrays. The default is True.
        perm : ndarray, optional
            Permutation indexes. The default is None.

        Raises
        ------
        Exception
            If the permutation indexes is wrong size.

        Returns
        -------
        list
            List of input/output data dicts.
        """
        
        data = self.get_data(split,numpy)
        X = data['x']
        y = data['y']
        size = len(y)
        
        if perm is None:
            perm = np.random.permutation(size)
        elif len(perm) != size:
            raise Exception("Permutation provided is wrong length: "+\
                            str(len(perm))+" vs "+str(size))
        
        X = X[perm,:]
        y = y[perm,:]
        
        x_folds = np.split(X,N,axis=0)
        y_folds = np.split(y,N,axis=0)
        return [{'x':x_folds[i],'y':y_folds[i]} for i in range(N)]

    def apply_fn(self,fn):
        """
        Applies a function mapping to each element in the feature data.

        Parameters
        ----------
        fn : function
            Mapping function for feature vectors.

        Returns
        -------
        None.
        """
        
        self.check_Data()
        for split,data_ in self.processed_data.items():
            x = data_['x']
            x = np.array([fn(xi) for xi in x])
            data_['x'] = x


class BinaryMLTMVectorizer(object):
    """
    Vectorizes a set of text (strings) with binary variables, based on the
    present ancestors in a tree for each word in the string.
    """
    
    def __init__(self,tree):
        """
        Parameters
        ----------
        tree : WordEmbedMLTM
            Modified latent tree model.
        """
        
        self.tree = tree
        self.nl = pd.Series(self.tree.non_leaves)
    
    def __len__(self):
        return len(self.nl)
    
    def __call__(self,data):
        """
        Calls vectorize(data).

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame with a 'Text' column.

        Returns
        -------
        data : pandas.DataFrame
            DataFrame with new columns, one for each binary latent variable.
        """
        
        return self.vectorize(data)
    
    def vectorize(self,text):
        """
        Converts a string of text into a numerical vector of features based
        on the word embedding LTM.

        Parameters
        ----------
        text : str
            input text.

        Returns
        -------
        ndarray
            numerical feature vector.
        """
        
        lv_active = set()
        words = word_tokenize(text)
        for word in words:
            if word in self.tree:
                ancestors = self.tree.word_ancestors(word)
                lv_active.update(ancestors)
                
        return self.nl.isin(lv_active).values


class NMTVectorizerWithMLTM(NMTVectorizer):
    """
    Extends NMTVectorizer by adding support for another data vector from the
    WordEmbedMLTM.
    """
    
    def __init__(self, source_vocab, target_vocab, max_source_length, 
                 max_target_length):
        """
        

        Parameters
        ----------
        source_vocab : SequenceVocabulary
            Maps source words to integers.
        target_vocab : SequenceVocabulary
            Maps target words to integers.
        max_source_length : int
            The longest sequence in the source dataset.
        max_target_length : int
            The longest sequence in the target dataset.
        mltm_vectorizer : BinaryMLTMVectorizer
            Vectorizes English text by latent variables.
        """
        
        super().__init__(source_vocab,target_vocab,max_source_length,
                         max_target_length)
        self.mltm_vectorizer = None
        
    def set_mltm_vectorizer(self,mltm_vectorizer):
        self.mltm_vectorizer = mltm_vectorizer

    def vectorize(self, source_text, target_text, use_dataset_max_lengths=True):
        """
        Calls super mehtod and adds a MLTM vector to the data dict.

        Parameters
        ----------
        source_text : str
            Text from the source language.
        target_text : str
            Text from the source language.
        use_dataset_max_lengths : bool, optional
            Whether to use the global max vector lengths. The default is True.

        Returns
        -------
        data : dict
            Vectorized data point as a dict.
        """
       
        data = super().vectorize(source_text, target_text, use_dataset_max_lengths)
        
        mltm_x_vector = self.mltm_vectorizer.vectorize(source_text.lower())
        mltm_x_vector = mltm_x_vector.astype(np.float32)
        
        data["x_source_mltm_vector"] = mltm_x_vector
        return data


class NMTDatasetWithMLTM(NMTDataset):
    
    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point: (x_data, y_target, class_index)
        """
        
        row = self._target_df.iloc[index]

        vector_dict = self._vectorizer.vectorize(row.source_language, row.target_language)

        return {"x_source": vector_dict["source_vector"], 
                "x_target": vector_dict["target_x_vector"],
                "y_target": vector_dict["target_y_vector"], 
                "x_source_length": vector_dict["source_length"],
                "x_source_mltm_vector": vector_dict["x_source_mltm_vector"]}




class MLPFactory(object):
    """
    Factory for generating MLP's with user-defined architecture.
    """
    
    def __init__(self,n_features,class_labels,hidden_sizes,activation="ReLU",
                 dropout=None):
        """
        Parameters
        ----------
        n_features : int
            Number of inputs.
        class_labels : list
            Labels for all classes.
        hidden_sizes : list
            List of hidden layer sizes.
        activation : str, optional
            Activation function for hidden nodes. The default is "ReLU".
        dropout : float, optional
            Dropout probability for hidden nodes. The default is None.

        Returns
        -------
        None.
        """
        
        self.n_features = n_features
        self.class_labels = np.array(class_labels)
        self.n_classes = len(class_labels)
        if np.isscalar(hidden_sizes):
            hidden_sizes = [hidden_sizes]
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
    
    def generate(self):
        """
        Generates a new MLP using the nn.Sequential class.

        Returns
        -------
        mlp : nn.Module
            An MLP with the user-defined architecture.
        """
    
        components = []
        components.append(nn.Linear(self.n_features,self.hidden_sizes[0]))
        self._activation(components,self.activation)
        self._dropout(components,self.dropout)
        
        for i in range(1,len(self.hidden_sizes)):
            components.append(nn.Linear(self.hidden_sizes[i-1],self.hidden_sizes[i]))
            self._activation(components,self.activation)
            self._dropout(components,self.dropout)
        
        components.append(nn.Linear(self.hidden_sizes[-1],self.n_classes))

        mlp = nn.Sequential(*components)
        
        num_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
        print("Created MLP with "+str(num_params)+" learnable params")
        
        return mlp
        
    def _activation(self,components,activation):
        """
        Creates a new activation function and adds it to the list of 
        components.

        Parameters
        ----------
        components : list
            component list for use in nn.Sequential.
        activation : str
            Activation function (Sigmoid, ReLU).

        Raises
        ------
        Exception
            If activation fn is unrecognized.

        Returns
        -------
        None.
        """
        
        if activation == "ReLU":
            components.append(nn.ReLU())
        elif activation == "Sigmoid":
            components.append(nn.Sigmoid())
        else:
            raise Exception("Invalid activation fn: "+activation)
        
    def _dropout(self,components,dropout=None):
        """
        Adds a dropout object to the list of components

        Parameters
        ----------
        components : list
            component list for use in nn.Sequential.
        dropout : float, optional
            Dropout probability. The default is None.

        Returns
        -------
        None.
        """
        
        if dropout is not None:
            components.append(nn.Dropout(dropout))


class NMTModelTrainer(object):
    """
    Trains an NMTModel with user-defined parameters. Adapted from Rao and
    McMahan (see PyTorchNLPBook_util.py).
    """
    
    def __init__(self,model,dataset,args):
        """
        Sets model, data, and training algo parameters.

        Parameters
        ----------
        model : NMTModel
            The neural machine translation model.
        dataset : NMTDataset
            Translation text data.
        args : Namespace
            Training parameters.
        """
        
        self.args = args
        self.dataset = dataset
        self.model = model.to(args.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                   mode='min', factor=0.5,
                                                   patience=1)
        
        vectorizer = dataset.get_vectorizer()
        self.mask_index = vectorizer.target_vocab.mask_index
        self.train_state = make_train_state(args)
        
        self.epoch_bar = tqdm_notebook(desc='training routine', 
                                  total=args.num_epochs,
                                  position=0)
        
        self.dataset.set_split('train')
        self.train_bar = tqdm_notebook(desc='split=train',
                                  total=dataset.get_num_batches(args.batch_size), 
                                  position=1, 
                                  leave=True)
        self.dataset.set_split('val')
        self.val_bar = tqdm_notebook(desc='split=val',
                                total=dataset.get_num_batches(args.batch_size), 
                                position=1, 
                                leave=True)

    def train(self):
        """
        Runs the training algorithm.

        Returns
        -------
        None.
        """
        
        args = self.args
        model = self.model
        dataset = self.dataset
        train_state = self.train_state
        optimizer = self.optimizer
        scheduler = self.scheduler
        train_bar = self.train_bar
        val_bar = self.val_bar
        epoch_bar = self.epoch_bar
        
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index
    
            # Iterate over training dataset
    
            running_loss,running_acc = self.train_loop(epoch_index, args, 
                                                       model, dataset, 
                                                       optimizer, train_bar)
    
            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)
            
            running_loss,running_acc = self.val_loop(epoch_index, args, model, 
                                                     dataset, optimizer, val_bar)
            
            
            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)
            
            print("Epoch "+str(epoch_index+1)+": Running loss="+ \
                          str(running_loss)+", Running Acc="+str(running_acc))
                
            train_state = update_train_state(args=args, model=model, 
                                             train_state=train_state)
    
            scheduler.step(train_state['val_loss'][-1])
            
            if train_state['stop_early']:
                break
                
            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.set_postfix(best_val=train_state['early_stopping_best_val'] )
            epoch_bar.update()
        
        state_dict = torch.load(train_state['model_filename'])
        model.load_state_dict(state_dict)
        return model
        
    def train_loop(self,epoch_index,args,model,dataset,optimizer,train_bar):
        """
        Runs the training for a single epoch.

        Parameters
        ----------
        epoch_index : int
            Index of the epoch.
        args : Namespace
            Training parameters.
        model : NMTModel
            Translation model to train.
        dataset : Dataset
            Torch dataset.
        optimizer : Optimizer
            The torch optimizer object.
        train_bar : tqdm_notebook
            Notebook for recording training stats.

        Returns
        -------
        running_loss : float
            Average loss for all training examples.
        running_acc : float
            Average accuracy on training examples.
        """
        
        dataset.set_split('train')
        batch_generator = generate_nmt_batches(dataset, 
                                               batch_size=args.batch_size, 
                                               device=args.device)
        
        running_loss = 0.0
        running_acc = 0.0
        model.train()

        for batch_index, batch_dict in enumerate(batch_generator):
            # the training routine is these 5 steps:

            # --------------------------------------    
            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            if isinstance(model,NMTModelWithMLTM):
                y_pred = model(batch_dict['x_source'], 
                           batch_dict['x_source_mltm_vector'],
                           batch_dict['x_source_length'], 
                           batch_dict['x_target'])
            else:
                y_pred = model(batch_dict['x_source'], 
                           batch_dict['x_source_length'], 
                           batch_dict['x_target'])

            # step 3. compute the loss
            loss = sequence_loss(y_pred, batch_dict['y_target'], self.mask_index)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()
            # -----------------------------------------
            # compute the running loss and running accuracy
            running_loss += (loss.item() - running_loss) / (batch_index + 1)

            acc_t = compute_accuracy(y_pred, batch_dict['y_target'], self.mask_index)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            
            # update bar
            train_bar.set_postfix(loss=running_loss, acc=running_acc, 
                            epoch=epoch_index)
            train_bar.update()
        
        return running_loss,running_acc

    def val_loop(self,epoch_index,args,model,dataset,optimizer,val_bar):
        """
        Evaluates the model on the validation set.

        Parameters
        ----------
        epoch_index : int
            Index of the epoch.
        args : Namespace
            Training parameters.
        model : NMTModel
            Translation model to train.
        dataset : Dataset
            Torch dataset.
        optimizer : Optimizer
            The torch optimizer object.
        val_bar : tqdm_notebook
            Notebook for recording validation stats.

        Returns
        -------
        running_loss : float
            Average loss for all validation examples.
        running_acc : float
            Average accuracy on validation examples.
        """
        
        dataset.set_split('val')
        batch_generator = generate_nmt_batches(dataset, 
                                               batch_size=args.batch_size, 
                                               device=args.device)
        running_loss = 0.0
        running_acc = 0.0
        model.eval()
        
        for batch_index, batch_dict in enumerate(batch_generator):
            # step 1. compute the output
            if isinstance(model,NMTModelWithMLTM):
                y_pred = model(batch_dict['x_source'], 
                           batch_dict['x_source_mltm_vector'],
                           batch_dict['x_source_length'], 
                           batch_dict['x_target'])
            else:
                y_pred = model(batch_dict['x_source'], 
                           batch_dict['x_source_length'], 
                           batch_dict['x_target'])

            # step 2. compute the loss
            loss = sequence_loss(y_pred, batch_dict['y_target'], self.mask_index)
            
            # -----------------------------------------
            # compute the running loss and running accuracy
            running_loss += (loss.item() - running_loss) / (batch_index + 1)

            acc_t = compute_accuracy(y_pred, batch_dict['y_target'], self.mask_index)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            
            # update bar
            val_bar.set_postfix(loss=running_loss, acc=running_acc, 
                            epoch=epoch_index)
            val_bar.update()
        
        return running_loss,running_acc

    def test(self):
        """
        Tests the model on the test set, measuring accuracy.

        Returns
        -------
        float
            Total accuracy of the model on the test set.
        """
        
        args = self.args
        model = self.model
        dataset = self.dataset
        
        dataset.set_split('test')
        batch_generator = generate_nmt_batches(dataset, 
                                               batch_size=len(dataset), 
                                               device=args.device)

        acc_sum = 0.0
        model.eval()
        
        for batch_index, batch_dict in enumerate(batch_generator):
            # step 1. compute the output
            if isinstance(model,NMTModelWithMLTM):
                y_pred = model(batch_dict['x_source'], 
                           batch_dict['x_source_mltm_vector'],
                           batch_dict['x_source_length'], 
                           batch_dict['x_target'])
            else:
                y_pred = model(batch_dict['x_source'], 
                           batch_dict['x_source_length'], 
                           batch_dict['x_target'])

            acc_t = compute_accuracy(y_pred, batch_dict['y_target'], self.mask_index)
            acc_sum += acc_t
        
        return acc_sum / (batch_index+1)


class SimpleModelTrainer(object):
    """
    Utility for training a PyTorch module.
    """
    
    def __init__(self,dataset,loss_fn,**kwargs):
        """
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            An object whose class extends a PyTorch Dataset. Must have a set_split
            method.
        loss_fn : function
            The loss function for gradient descent training.
        **kwargs : kwargs
            Options for training: epochs, lr, batch_size.

        Returns
        -------
        None.
        """
        
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.epochs = kwargs.get('epochs',100)
        self.lr = kwargs.get('lr',1e-3)
        self.batch_size = kwargs.get('batch_size',64)
        self.early_stopping_criteria = kwargs.get('early_stopping_criteria',np.inf)

    def train_model(self,model):
        """
        Runs a training procedure on a PyTorch module using the dataset and
        loss function.
    
        Parameters
        ----------
        model : nn.Module
            Model to train (PyTorch module).

        Returns
        -------
        best_model
            The model with the best validation loss score.
        train_state : dict
            information about the training sequence.
        """
        
        train_state = {'stop_early': False,
                'early_stopping_step': 0,
                'early_stopping_best_val': 1e8,
                'learning_rate': self.lr,
                'epoch_index': 0,
                'train_loss': [],
                'val_loss': [],
                'best_model':model}
        
        dataset = self.dataset
        loss_fn = self.loss_fn
        
        dataset.set_split('train')
        print("Training module with "+str(len(dataset))+" examples")
        
        data_loader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True,
                                 drop_last=True)
        
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            train_state['epoch_index'] = epoch
            #First step in each epoch is to train over all batches
            model.train()
            dataset.set_split('train')
            train_loss = 0
            for b_i,batch_data in enumerate(data_loader):
                #Step 1: zero gradients
                optimizer.zero_grad()
                #Step 2: run forward
                X = batch_data['x']
                output = model(X)
                #Step 3: compute loss
                target = batch_data['y']
                loss = loss_fn(output,target)
                #Step 4: run backward
                loss.backward()
                #Step 5: update
                optimizer.step()
                
                #Record accumulated loss
                new_loss = loss.item()
                train_loss += new_loss
    
            train_loss /= b_i
            train_state['train_loss'].append(train_loss)
            
            #After training, compute loss on validation set and check for early stop
            model.eval()
            dataset.set_split('val')
            val_loss = 0
            for b_i,batch_data in enumerate(data_loader):
                #Step 1: run forward
                X = batch_data['x']
                output = model(X)
                #Step 2: compute loss
                target = batch_data['y']
                loss = loss_fn(output,target)
                
                #Record accumulated loss
                new_loss = loss.item()
                val_loss += new_loss
            
            val_loss /= b_i
            train_state['val_loss'].append(val_loss)
            
            print("Finished epoch "+str(epoch+1)+". Train loss="+\
                          str(train_loss)+", Val loss="+str(val_loss))
            
            if val_loss < train_state['early_stopping_best_val']:
                #new best model, reset stopping counter, store model
                train_state['early_stopping_step'] = 0
                train_state['early_stopping_best_val'] = val_loss
                best_model = copy.deepcopy(model)
                best_model.load_state_dict(model.state_dict())
                train_state['best_model'] = best_model
            else:
                #val loss not improved; increase early stopping counter
                train_state['early_stopping_step'] += 1
                if train_state['early_stopping_step'] >= self.early_stopping_criteria:
                    train_state['stop_early'] = True
                    print("Val loss failed to improve. Stopping early.")
                    break
            
        return train_state['best_model'],train_state


class NMTSamplerWithMLTM(NMTSampler):
    """
    Extends the NMTSampler class to handle NMTModelWithMLTM models.
    """
    
    def __init__(self, vectorizer, model):
        """
        Parameters
        ----------
        vectorizer : object
            Vectorizes text data.
        model : NMTModel
            Trained neural machined translation model.

        Returns
        -------
        None.
        """
        
        super().__init__(vectorizer,model)
    
    def apply_to_batch(self, batch_dict):
        """
        Generates predictions and attentions for a batch.

        Parameters
        ----------
        batch_dict : dict
            Map of input data Tensors.

        Returns
        -------
        None.
        """
        
        self._last_batch = batch_dict
        
        if isinstance(self.model,NMTModelWithMLTM):
            y_pred = self.model(x_source=batch_dict['x_source'], 
                                x_mltm=batch_dict['x_source_mltm_vector'],
                            x_source_lengths=batch_dict['x_source_length'], 
                            target_sequence=batch_dict['x_target'])
        else:
            y_pred = self.model(x_source=batch_dict['x_source'], 
                            x_source_lengths=batch_dict['x_source_length'], 
                            target_sequence=batch_dict['x_target'])
        self._last_batch['y_pred'] = y_pred
        
        attention_batched = np.stack(self.model.decoder._cached_p_attn).transpose(1, 0, 2)
        self._last_batch['attention'] = attention_batched


def split_data(text_df,splits=None,rand_perm=True):
    """
    Splits a DataFrame into 3 distinct DataFrames based on the given percentages
    and returns a dict of the data.

    Parameters
    ----------
    text_df : pd.DataFrame
        DataFrame with text data.
    splits : dict, optional
        The split percentages. The default is None.
    rand_perm : bool, optional
        Permutation indexes. The default is True.

    Raises
    ------
    Exception
        Raised if split percentages don't sum to unity.

    Returns
    -------
    all_data : dict
        dict of data by split.
    """
    
    if splits is None:
        splits = {'train':0.6,'val':0.1,'test':0.3}
    
    if np.round(np.sum(list(splits.values())),4) != 1:
        raise Exception("Split percentages do not sum to 1")
    
    size = len(text_df)
    if rand_perm:
        perm_idx = np.random.permutation(size)
    else:
        perm_idx = np.arange(size)
    
    text_df = text_df.iloc[perm_idx,:]
    
    all_data = dict()
    keys = list(splits.keys())
    pct = list(splits.values())
    count = np.round(np.array(pct) * size).astype(np.int32)
    split_idx = np.cumsum(count)[:-1]
    data_list = np.split(text_df,split_idx,axis=0)
    all_data = {keys[i]:data for i,data in enumerate(data_list)}

    return all_data

def classify(dataset,classifier,feat_mask=None):
    """
    Performs a standard classification test with the given classifier.

    Parameters
    ----------
    dataset : TextDataset
        TextDataset with train/val/test splits.
    classifier : object
        Classifier must have fit(X,y) & predict(X) methods.

    Returns
    -------
    acc : float
        % Accuracy.
    y_test : ndarray
        class labels.
    pred : ndarray
        predicted class labels.

    """
    
    train = dataset.get_data('train',True)
    X_train = train['x']
    if feat_mask is not None:
        X_train = X_train[:,feat_mask]
    y_train = train['y']
    
    classifier.fit(X_train,y_train)
    
    test = dataset.get_data('test',True)
    X_test = test['x']
    if feat_mask is not None:
        X_test = X_test[:,feat_mask]
    y_test = test['y']
    
    pred = classifier.predict(X_test)
    
    acc = np.count_nonzero(pred==y_test) / len(y_test)
    return acc,y_test,pred


def filter_nmt_file(filename,filter_fn=None):
    """
    Reads a English -> French text file and filters the lines based on the
    given filter_fn. If filter_fn is None, the default filter will be
    used: 
        include only those english sentences that start with:
            'i am'
            'he is'
            'she is'
            'they are'
            'you are'
            'we are'

    Parameters
    ----------
    filename : str
        Name of text file.
    filter_fn : function
        Determines which sentence pairs will be included

    Returns
    -------
    filtered_lines : list
        List of strings of English -> French text.
    """
    
    if filter_fn is None:
        filter_fn = lambda en : en.lower().startswith('i am') or \
            en.lower().startswith('he is') or \
            en.lower().startswith('she is') or \
            en.lower().startswith('they are') or \
            en.lower().startswith('you are') or \
            en.lower().startswith('we are')
    
    filtered_lines = []
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            text = line.split('\t')
            en = text[0]
            fra = text[1]
            if filter_fn(en):
                filtered_lines.append(en.lower() + '\t' + fra.lower())

    return filtered_lines

def create_nmt_data(text,train_pct=0.7,val_pct=0.15):
    """
    Given a list of lines of English/French text, creates a DataFrame
    with train/val/test split labels.

    Parameters
    ----------
    text : list
        List of English/French lines split by tabs.

    Returns
    -------
    text_df : pd.DataFrame
        DF with columns: source_language, target_language, split.
    """
    
    if train_pct + val_pct >= 1:
        raise Exception("train_pct + val_pct must be < 1.0")
        
    source = []
    target = []
    for line in text:
        text = line.split('\t')
        source.append(text[0])
        target.append(text[1])
    
    text_df = pd.DataFrame({'source_language':source,'target_language':target})
    text_df['split'] = 'train'
    text_df = text_df.sample(frac=1).reset_index(drop=True)
    idx = int(len(text_df)*train_pct)
    text_df.loc[:idx,'split'] = 'train'
    idx2 = idx + int(len(text_df)*val_pct)
    text_df.loc[idx:idx2,'split'] = 'val'
    text_df.loc[idx2:,'split'] = 'test'
    
    return text_df

def process_glove_data(filename):
    """
    Reads a glove word embedding text file and generates a DataFrame with
    the embeddings.

    Parameters
    ----------
    filename : str
        Name of embeddings text file.

    Returns
    -------
    embed_df : pd.DataFrame
        DF with words as the index and embedding dimensions as columns.
    """

    word_list = []
    embed_list = []
    with open(filename,encoding="utf8") as file:
        lines = file.readlines()
        for line in lines:
            toks = line.split(' ')
            word_list.append(toks[0])
            vec = [float(tok) for tok in toks[1:]]
            embed_list.append(vec)
    
    embed = np.array(embed_list,dtype=float)
    embed_df = pd.DataFrame(embed,index=word_list)
    embed_df.index = embed_df.index.str.lower()
    
    return embed_df

def get_pretrained_embeddings(source_vocab,embed_df):
    """
    Creates a Tensor for use as an Embedding initialization from the source
    vocabulary and pre-defined word embeddings.

    Parameters
    ----------
    source_vocab : Vocabulary
        Vocabulary with lookups for source language.
    embed_df : pd.DataFrame
        Word embeddings with columns as dimensions and words as index.

    Returns
    -------
    embed_tensor : torch.Tensor
        Word embeddings as Tensor.
    """
    
    num_tokens = len(source_vocab)
    embedding_dim = embed_df.shape[1]
    weights = np.zeros((num_tokens,embedding_dim),dtype=np.float32)
    
    for idx in range(num_tokens):
        token = source_vocab.lookup_index(idx)
        if token in embed_df.index:
            weights[idx,:] = embed_df.loc[token]
        else:
            weights[idx,:] = np.random.randn(1,embedding_dim)
    
    embed_tensor = torch.FloatTensor(weights)
    return embed_tensor

def eval_nmt_bleu(model,dataset,vectorizer,args):
    """
    Evaluates the trained model on the test set using the bleu_score method
    from NLTK.

    Parameters
    ----------
    model : NMTModel
        Trained NMT model.
    dataset : Dataset
        Dataset with Source/Target sentences.
    vectorizer : object
        Vectorizes sentences.
    args : Namespace
        Simulation parameters.

    Returns
    -------
    float
        Average bleu-4 score.
    bleu4 : TYPE
        Array of sentence bleu-4 scores.
    """
    
    model = model.eval().to(args.device)
    
    sampler = NMTSamplerWithMLTM(vectorizer, model)
    
    dataset.set_split('test')
    batch_generator = generate_nmt_batches(dataset, 
                                           batch_size=args.batch_size, 
                                           device=args.device)
    
    test_results = []
    for batch_dict in batch_generator:
        sampler.apply_to_batch(batch_dict)
        for i in range(args.batch_size):
            test_results.append(sampler.get_ith_item(i, False))
    
    bleu4 = np.array([r['bleu-4'] for r in test_results])*100
    return np.mean(bleu4),bleu4



