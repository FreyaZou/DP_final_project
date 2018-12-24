import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from itertools import chain
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss, classification_report, roc_auc_score

import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Input, Embedding, Bidirectional, LeakyReLU,concatenate, GRU, Flatten, LSTM, Conv1D, MaxPooling1D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
import os

with open("./train_text.p", "rb") as input_file:
    train_text = pickle.load(input_file)
with open("./test_text.p", "rb") as input_file:
    test_text = pickle.load(input_file)
with open("./train_meta.p", "rb") as input_file:
    train_meta = pickle.load(input_file)
with open("./test_meta.p", "rb") as input_file:
    test_meta = pickle.load(input_file)
with open("./Y_train.p", "rb") as input_file:
    Y_train = pickle.load(input_file)
with open("./Y_test.p", "rb") as input_file:
    Y_test = pickle.load(input_file)


class Bidir_LSTM:
    def __init__(self, train_text, test_text, train_meta, test_meta, y_train, y_test):
        self.train_text = np.array(train_text)
        self.test_text = np.array(test_text)
        self.y_train = y_train
        self.y_test = y_test
        self.max_len = 300
        self.train_meta_features = train_meta
        self.test_meta_features = test_meta
        
        self.train_text_new = self.preprocess_text(self.train_text)
        self.test_text_new = self.preprocess_text(self.test_text)
        self.train_sequence, self.vocab_dict = self.create_dict(self.train_text, training = True)
        self.test_sequence = self.word2seq(self.test_text)
        
        self.embedding_matrix = self.glove_embedding()
        self.model = self.train_model()
        self.pred_train = self.predict([self.train_sequence, self.train_meta_features])
        self.train_AUC = self.auroc(self.y_train, self.pred_train)
        self.pred_test = self.predict([self.test_sequence, self.test_meta_features])
        self.test_AUC = self.auroc(self.y_test, self.pred_test)
    
    def preprocess_text(self, all_sentences):
        def clean_text(text):
            ## Remove puncuation
            text = text.translate(string.punctuation)
            
            ## Convert words to lower case and split them
            text = text.lower().split()
            
            ## Remove stop words
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops and len(w) >= 3]
            
            text = " ".join(text)
            
            text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
            text = re.sub(r"what's", "what is ", text)
            text = re.sub(r"\'s", " ", text)
            text = re.sub(r"\'ve", " have ", text)
            text = re.sub(r"n't", " not ", text)
            text = re.sub(r"i'm", "i am ", text)
            text = re.sub(r"\'re", " are ", text)
            text = re.sub(r"\'d", " would ", text)
            text = re.sub(r"\'ll", " will ", text)
            text = re.sub(r",", " ", text)
            text = re.sub(r"\.", " ", text)
            text = re.sub(r"!", " ! ", text)
            text = re.sub(r"\/", " ", text)
            text = re.sub(r"\^", " ^ ", text)
            text = re.sub(r"\+", " + ", text)
            text = re.sub(r"\-", " - ", text)
            text = re.sub(r"\=", " = ", text)
            text = re.sub(r"'", " ", text)
            text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
            text = re.sub(r":", " : ", text)
            text = re.sub(r" e g ", " eg ", text)
            text = re.sub(r" b g ", " bg ", text)
            text = re.sub(r" u s ", " american ", text)
            text = re.sub(r"\0s", "0", text)
            text = re.sub(r" 9 11 ", "911", text)
            text = re.sub(r"e - mail", "email", text)
            text = re.sub(r"j k", "jk", text)
            text = re.sub(r"\s{2,}", " ", text)
            
            return text
        
        text_new = []
        for line in all_sentences:
            text_new_ = clean_text(line)
            text_new.append(text_new_)
        return text_new

    def create_dict(self, text_new, training = True):
        vocabulary_size = 100000
        
        tokenizer = Tokenizer(num_words= vocabulary_size,filters='"#*$%&()+,-./:;<=>@[\]^_`{|}~',
                              lower=True, split=' ', char_level=False, oov_token='*UNK*')
            
        tokenizer.fit_on_texts(text_new)
        sequences = tokenizer.texts_to_sequences(text_new)
        padded_sequences = pad_sequences(sequences, maxlen = self.max_len)

        if training:
            vocab_dict = tokenizer.word_index
            return padded_sequences, vocab_dict
        else:
            return padded_sequences

    def word2seq(self, test_text):
#        vocabulary_size = 100000
#        tokenizer = Tokenizer(num_words= vocabulary_size,filters='"#*$%&()+,-./:;<=>@[\]^_`{|}~',
#                              lower=True, split=' ', char_level=False, oov_token='*UNK*')
#
#        clean_text = tokenizer.fit_on_texts(test_text)
        text_id = []
        
        for line in test_text:
            token_id = []
        
            for index, token in enumerate(line.split()):
                if token in self.vocab_dict:
                    token_id.append(self.vocab_dict.get(token))
                else:
                    token_id.append(self.vocab_dict.get('*UNK*'))
            text_id.append(token_id)
        padded_sequences = pad_sequences(text_id, maxlen = self.max_len)
        return padded_sequences


    def glove_embedding(self):
        embeddings_index = dict()
        f = open('glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        
        
        embedding_matrix = np.zeros((len(self.vocab_dict) + 1, 100))
        for word, i in self.vocab_dict.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix


    def train_model(self):
        
        tokens_input = Input(shape=[self.max_len], name="SentencesTokens")
        meta_input = Input(shape=(6,), name='meta_input')
        # load pre-trained word embeddings into an Embedding layer
        
        emb = Embedding(len(self.vocab_dict) + 1, 100, input_length = self.max_len,
                        weights=[self.embedding_matrix], trainable=True) (tokens_input)
          
        rnn_outputs1 = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3,return_sequences=True))(emb)
        rnn_outputs = GRU(128, activation='relu',recurrent_dropout=0.3,return_sequences=False)(rnn_outputs1)#‘relu'
        ## attention (len(rnn_outputs shape) = 3)
        # att = Attention(max_len)(rnn_outputs)
        # reshape = Dense(240, activation="relu")(att)
        
        aggregate_vectors = concatenate([rnn_outputs, meta_input])

        h1 = Dropout(0.3)(aggregate_vectors)
        h2 = Dense(64, activation='relu')(h1)
        outputs  = Dense(1, activation='sigmoid')(h2)
        self.model = Model(inputs=[tokens_input , meta_input], outputs=[outputs])
        
        adam_opt = Adam(lr = 0.001)
        self.model.compile(loss='binary_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
        self.model.fit([self.train_sequence, self.train_meta_features], self.y_train, batch_size=128,
                       epochs=2, validation_split=0.3,
                       callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])
        self.model.save('dir_LSTM_.h5')
        return self.model


    def predict(self,inputs):
        return self.model.predict(inputs)
    
    def evaluate(self):
        return self.model.evaluate([self.test_sequence, self.test_meta_features], self.y_test)
    
    def auroc(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)


print('==============================Birdir_LSTM model===============================')
Bidir_LSTM_model = Bidir_LSTM(train_text, test_text, train_meta, test_meta, Y_train, Y_test)
#Bidir_LSTM_model.train_model()

print('Test evaluation-- Loss: %3f, Accuracy: %3f'%tuple(Bidir_LSTM_model.evaluate()))
print('training AUC: %3f, test AUC: %3f'%tuple([Bidir_LSTM_model.train_AUC, Bidir_LSTM_model.test_AUC]))
