# -*- coding: utf-8 -*-

import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
le = LabelEncoder()


'''
Path to biobert checkpoints, which are already converted to pytorch
'''
root_dir = "./"
path_=os.path.join(root_dir,'biobert_T')
VOCAB_FILE=path_+'/vocab.txt'
print(VOCAB_FILE)
tokenizer = BertTokenizer(vocab_file=VOCAB_FILE, do_lower_case=True)
'''
The model class, which extracts data, process it and convert to required formats for BERT modelling,
Train the model and save it
Test for future predictions and compute the F-scores
'''

class Model:
    def __init__(self,path):
        # self.args = args
        self.path=path
        self.MAX_LEN=128
       
        # if not os.path.isdir(self.opath):
        #     os.makedirs(self.opath)
            
            
    def extract_data(self,name):
        file =self.path+name
        df = pd.read_csv(file, delimiter='\t', header=None, names=['pmid','sentence', 'entity_pair','label','path','triplets','types'])
        df.replace(np.nan,'NIL', inplace=True)
        
        sentences = df.sentence.values
        entity=df.entity_pair.values
        labels = df.label.values
        paths=df.path.values
        triplets=df.triplets.values
        types=df.types.values
        
        return (sentences,entity,labels,paths,triplets,types)
    def process_inputs(self,e,t,sentences,labels,fl=1):
      entity=[ast.literal_eval(x) for x in e]
      triplets=[ast.literal_eval(x) for x in t]
      triplets_sents=[('_'.join(x[0:2]),'_'.join(x[2:4])) if len(x)==4 else ('_'.join(x[0:2]),'_'.join(x[1:3])) for x in triplets]
      if fl==1:
        sentences = [tokenizer.encode_plus(sent,triplets_sents[i],add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]
      else:
        sentences= [tokenizer.encode_plus(sent,entity[i],add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]
      
      tags_vals = list(labels)
      le.fit(labels)
      le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
      labels=le.fit_transform(labels)
      # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
      input_ids = [inputs["input_ids"] for inputs in sentences]

      # Pad our input tokens
      input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")
      attention_masks = []

      # Create a mask of 1s for each token followed by 0s for padding
      for seq in input_ids:
        seq_mask= [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

        
      token_type_ids=[inputs["token_type_ids"] for inputs in sentences]
      token_type_ids=pad_sequences(token_type_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")

      inputs, labels,types = input_ids, labels,token_type_ids
      masks,_= attention_masks, input_ids
      # Convert all of our data into torch tensors, the required datatype for our model

      self.inputs = torch.tensor(inputs).to(torch.int64)
      # validation_inputs = torch.tensor(validation_inputs).to(torch.int64)
      self.labels = torch.tensor(labels).to(torch.int64)
      # validation_labels = torch.tensor(validation_labels).to(torch.int64)
      self.masks = torch.tensor(masks).to(torch.int64)
      # validation_masks = torch.tensor(validation_masks).to(torch.int64)
      self.types=torch.tensor(types).to(torch.int64)
      self.data = TensorDataset(self.inputs, self.types,self.masks, self.labels)
      self.sampler = RandomSampler(self.data)
      self.dataloader = DataLoader(self.data, sampler=self.sampler, batch_size=32)

      # return (self.inputs,self.labels,self.masks,self.types)
    def train_save_load(self,path_,train=1):
     
      self.model = BertForSequenceClassification.from_pretrained(path_, num_labels=2)
      self.model.cuda()
      param_optimizer = list(self.model.named_parameters())
      no_decay = ['bias', 'gamma', 'beta']
      optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                                                                                                                                     'weight_decay_rate': 0.0}]
      optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5)
      
      WEIGHTS_NAME = "CDR_model.bin"
      OUTPUT_DIR = root_dir+'./BIORELATION/CDR/'
      output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
      train_loss_set = []
      epochs = 10
      import time
      start_time = time.time()
      if train==1:
        for _ in trange(epochs, desc="Epoch"):
          # Trainin
          # Set our model to training mode (as opposed to evaluation mode
          self.model.train()
          # Tracking variables
          tr_loss = 0
          nb_tr_examples, nb_tr_steps = 0, 0
          # Train the data for one epoch
          for step, batch in enumerate(self.dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_token_type,b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            # loss = model(b_input_ids, token_type_ids=b_types, attention_mask=b_input_mask, labels=b_labels)
            loss,logits= self.model(b_input_ids, token_type_ids=b_token_type, attention_mask=b_input_mask, labels=b_labels)

            train_loss_set.append(loss.item())    
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
          print("Train loss: {}".format(tr_loss/nb_tr_steps))
        print("--- %s seconds ---" % (time.time() - start_time)) 
        torch.save(self.model.state_dict(), output_model_file)
      else:
        state_dict = torch.load(output_model_file)
        self.model.load_state_dict(state_dict) 
    def test(self):
      # Put model in evaluation mod
      self.model.eval()
      # Tracking variables 
      self.predictions , self.true_labels = [], []
      # Predict 
      for batch in self.dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids,b_type, b_input_mask, b_labels = batch
        # # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = self.model(b_input_ids, token_type_ids=b_type, attention_mask=b_input_mask)
        logits=outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Store predictions and true labels
        self.predictions.append(logits)
        self.true_labels.append(label_ids)
    def compute(self):
      flat_true_labels1=[]
      # Flatten the predictions and true values 
      flat_predictions = [item for sublist in self.predictions for item in sublist]
      flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
      flat_true_labels = [item for sublist in self.true_labels for item in sublist]
      print('Classification Report')
      labels_test=flat_predictions
      print(classification_report(flat_true_labels,flat_predictions))
      print(confusion_matrix(flat_true_labels,flat_predictions))
'''
The path where data is stored
'''
path=root_dir+'./BIORELATION/CDR/CDR_DATA/'

m = Model(path)
sentences_train,entity_train,labels_train,paths_train,triplets_train,types_train=m.extract_data('train1.tsv')
sentences_dev,entity_dev,labels_dev,paths_dev,triplets_dev,types_dev=m.extract_data('dev1.tsv')
TV=1#use train+Validation set else set TV!=1
if TV:
  sentences_train=np.append(sentences_train,sentences_dev)
  entity_train=np.append(entity_train,entity_dev)
  labels_train=np.append(labels_train,labels_dev)
  triplets_train=np.append(triplets_train,triplets_dev)
  types_train=np.append(types_train,types_dev)

print(len(sentences_train),len(entity_train),len(labels_train),len(triplets_train),len(types_train))

m.process_inputs(entity_train,triplets_train,sentences_train,labels_train,fl=1)

path_=os.path.join(root_dir,'biobert_T')
#Train the model from scratch
m.train_save_load(path_,train=1)

#to load the model directly set train!=1 else train=1
m.train_save_load(path_,train=0)

sentences_test,entity_test,labels_test,paths_test,triplets_test,types_test=m.extract_data('test1.tsv')

m.process_inputs(entity_test,triplets_test,sentences_test,labels_test,fl=1)

m.test()

m.compute()

