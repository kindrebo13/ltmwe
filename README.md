This project builds latent tree models using word embeddings obtained from Stanford NLP's GloVe project. The tree models are constructed using sklearn Agglomerative Clustering algorithm.

main_agn.py runs a text classfication test, comparing naive Bayes classification with a MLP classifier using the latent variable features from the tree model. The text classification data comes from https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv, and has been filtered and preprocessed. The preprocessed data is available in data/agn as a pickle file.
