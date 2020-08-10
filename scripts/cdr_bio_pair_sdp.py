# -*- coding: utf-8 -*-



import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# BERT imports
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import  BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import ast
# % matplotlib inline
import os
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
MAX_LEN=128

'''
path where BioBERT pretrained weights are stored
'''
path_=os.path.join(root_dir,'biobert')
VOCAB_FILE=path_+'/vocab.txt'
print(VOCAB_FILE)
tokenizer = BertTokenizer(vocab_file=VOCAB_FILE, do_lower_case=True)

df = pd.read_csv("train1", delimiter='\t', header=None, names=['pmid','sentence', 'entity_pair','label','path','triplets','types'])
# df = pd.read_csv("CHEMPROT_Train_GC.tsv", delimiter='\t', header=None, names=['sentence','label'])
df.replace(np.nan,'NIL', inplace=True)

print(df.head())

# Create sentence and label lists

sentences1 = df.sentence.values
print(len(sentences1))

entity=df.entity_pair.values
labels = df.label.values
paths=df.path.values
triplets_train=df.triplets.values
types=df.types.values
p=[x for x in labels if x==1]
n=[x for x in labels if x==0]
print(len(p),len(n))


c=ast.literal_eval(entity[0])
c1=ast.literal_eval(triplets_train[2])


df2 = pd.read_csv("dev1", delimiter='\t', header=None, names=['pmid','sentence', 'entity_pair','label','path','triplets','types'])
# df = pd.read_csv("CHEMPROT_Train_GC.tsv", delimiter='\t', header=None, names=['sentence','label'])
df2.replace(np.nan,'NIL', inplace=True)

# Create sentence and label lists
p_sentences=[]
sentences11 = df2.sentence.values
print(len(sentences11))

entity2=df2.entity_pair.values
labels2 = df2.label.values
path2=df2.path.values
triplet2=df2.triplets.values
types2=df2.types.values

p_sentences1=[(x,entity2[i],labels2[i],path2[i],triplet2[i],types2[i]) for i,x in enumerate(sentences11) if labels2[i]==1]
p_sentences2=[(x,entity2[i],labels2[i],path2[i],triplet2[i],types2[i]) for i,x in enumerate(sentences11) if labels2[i]==0]
# p_sentences2=p_sentences2[:len(p_sentences1)-1128]
# p_sentences=p_sentences1+p_sentences2
p_sentences=p_sentences1+p_sentences2
#p_sentences=[(x,sentences22[i],path_lengths_d[i],path_words_d[i],labels2[i]) for i,x in enumerate(sentences11) ]
p2=[x for x in labels2 if x==1]
n2=[x for x in labels2 if x==0]
print(len(p2),len(n2))
print(len(p_sentences))

s1=[x[0] for x in p_sentences]
# s2=[x[1] for x in p_sentences]
# pld=[x[2] for x in p_sentences]
# plw=[x[3] for x in p_sentences]
#l1==[x[4] for x in p_sentences]
e=[x[1] for x in p_sentences]
w1=[x[2] for x in p_sentences]
p=[x[3] for x in p_sentences]
tr=[x[4] for x in p_sentences]
ty=[x[5] for x in p_sentences]

print(len(w1),len(s1),len(e),len(p),len(tr),len(ty))



sentences1=np.append(sentences1,s1)
print(len(sentences1))
labels=np.append(labels,w1)
entity=np.append(entity,e)
triplets_train=np.append(triplets_train,tr)
types=np.append(types,ty)
# path_lengths=np.append(path_lengths,pld)
# path_words=np.append(path_words,plw)
paths=np.append(paths,p)
sentences_append=sentences1


triplets_append=[ast.literal_eval(x) for x in triplets_train]

triplets_sents=[('_'.join(x[0:2]),'_'.join(x[2:4])) if len(x)==4 else ('_'.join(x[0:2]),'_'.join(x[1:3])) for x in triplets_append]
triplets_sents[0]

print(len(sentences_append))


sentences_train= [tokenizer.encode_plus(sent,add_special_tokens=True, max_length=MAX_LEN) for i,sent in enumerate(sentences_append)]



# We need to add special tokens at the beginning and end of each sentence for BERT to work properly


# sentences_train = [tokenizer.encode_plus(entity[i],sent,add_special_tokens=True, max_length=MAX_LEN) for i,sent in enumerate(sentences_append)]#(pair position reversed)
# sentences_train = [tokenizer.encode_plus(sent,triplets_sents[i],add_special_tokens=True, max_length=MAX_LEN) for i,sent in enumerate(sentences_append)]
# sentences_train = [tokenizer.encode_plus(sent,add_special_tokens=True, max_length=MAX_LEN) for i,sent in enumerate(sentences_append)]

# sentences_train1 = ["[CLS] " + x[0] + " [SEP]" + x[1:-1] + " [SEP]"+x[-1] + " [SEP]" for x in triplets_append]


#tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences_train1]
#input_ids_train1 = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

#input_ids_train1 = pad_sequences(input_ids_train1, maxlen=MAX_LEN,truncating="post", padding="post")





labels_train = labels

tags_vals = list(labels)
# print(tags_vals[:10])
print('Analyze labels')
le.fit(labels_train)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

labels_train=le.fit_transform(labels_train)
print(labels_train[0])
# print(set(labels_train))

print(len(sentences_train))
print(len(labels_train))

len(labels_train)

if -1 in labels_train:
  print('yes')

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
# input_ids_train = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
input_ids_train = [inputs["input_ids"] for inputs in sentences_train]

# Pad our input tokens
input_ids_train = pad_sequences(input_ids_train, maxlen=MAX_LEN,truncating="post", padding="post")
attention_masks_train = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids_train:
  seq_mask_train = [float(i>0) for i in seq]
  attention_masks_train.append(seq_mask_train)

  
token_type_ids_train=[inputs["token_type_ids"] for inputs in sentences_train]
token_type_ids_train=pad_sequences(token_type_ids_train, maxlen=MAX_LEN,truncating="post", padding="post")
  
train_inputs, train_labels,train_type = input_ids_train, labels_train,token_type_ids_train
train_masks,_= attention_masks_train, input_ids_train
# Convert all of our data into torch tensors, the required datatype for our model

train_inputs = torch.tensor(train_inputs).to(torch.int64)
# validation_inputs = torch.tensor(validation_inputs).to(torch.int64)
train_labels = torch.tensor(train_labels).to(torch.int64)
# validation_labels = torch.tensor(validation_labels).to(torch.int64)
train_masks = torch.tensor(train_masks).to(torch.int64)
# validation_masks = torch.tensor(validation_masks).to(torch.int64)
train_type=torch.tensor(train_type).to(torch.int64)
batch_size = 32

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_type,train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained(path_, num_labels=2)

model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# This variable contains all of the hyperparemeter information our training loop needs
optimizer = BertAdam(optimizer_grouped_parameters,lr=2e-5,warmup=.1)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 10

import time
start_time = time.time()
for _ in trange(epochs, desc="Epoch"):
  
  
  # Training
  
  # Set our model to training mode (as opposed to evaluation mode)
  model.train()
  
  # Tracking variables
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  
  # Train the data for one epoch
  for step, batch in enumerate(train_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # batch = tuple(t for t in batch)
    # Unpack the inputs from our dataloader
    # b_input_ids,b_types,b_input_mask, b_labels = batch
    
    b_input_ids, b_token_type,b_input_mask, b_labels = batch
      # Clear out the gradients (by default they accumulate)
    optimizer.zero_grad()
      # Forward pass
      # loss = model(b_input_ids, token_type_ids=b_types, attention_mask=b_input_mask, labels=b_labels)
    loss = model(b_input_ids, token_type_ids=b_token_type, attention_mask=b_input_mask, labels=b_labels)
    
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





# If we save using the predefined names, we can load using `from_pretrained`

# model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
model_to_save=model
# CONFIG_NAME = "config1.json"
WEIGHTS_NAME = "BIO_PAIR_SDP.bin"

# TASK_NAME = 'BIOREL'
import os
OUTPUT_DIR = './BIORELATION'
output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
# output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)
print(output_model_file)

torch.save(model_to_save.state_dict(), output_model_file)


# df1 =pd.read_csv("TEST_FINAL_TUNED_nw.tsv", delimiter='\t', header=None, names=['sentence', 'entity_pair','label_t','types'])
df1 =pd.read_csv("test1.tsv", delimiter='\t', header=None, names=['pmid','sentence', 'entity_pair','label','path','triplets','types'])
# df1 =pd.read_csv("tuning_test1_mod.tsv", delimiter='\t', header=None, names=['sentence', 'entity_pair','label_t','types'])
# df1 =pd.read_csv("Modelstn.tsv", delimiter='\t', header=None, names=['sentence', 'entity_pair','label_p','label_t'])
# df1 = pd.read_csv("CHEMPROT_Test_GC.tsv", delimiter='\t', header=None, names=['sentence', 'label'])
df1.replace(np.nan,'NIL', inplace=True)

df1.head()

# Create sentence and label lists
# {0:3,1:4,2:5,3:6,4:9,'NIL':5}id-label
sentences1 = df1.sentence.values

entity1=df1.entity_pair.values

labels1 = df1.label.values
path1=df1.path.values
triplets1=df1.triplets.values
types1=df1.types.values
p=[x for x in labels1 if x==1]
n=[x for x in labels1 if x==0]
print(len(p),len(n))

entity1=[ast.literal_eval(x) for x in entity1]
entity1=[x[0]+' '+x[1] for x in entity1]


print(len(entity1),len(sentences1),len(labels1),len(path1))


MAX_LEN=128



import ast
triplets_test=[ast.literal_eval(x) for x in triplets1]
triplets_sents1=[('_'.join(x[0:2]),'_'.join(x[2:4])) if len(x)==4 else ('_'.join(x[0:2]),'_'.join(x[1:3])) for x in triplets_test]


sentences_test = [tokenizer.encode_plus(sent,triplets_sents1[i],add_special_tokens=True, max_length=MAX_LEN) for i,sent in enumerate(sentences1)]

print(labels1[:10])
print('Analyze labels test')
# We need to add special tokens at the beginning and end of each sentence for BERT to work properly
# sentences1 = ["[CLS] " + sentence + " [SEP]" for sentence in sentences1]


le.fit(labels1)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

labels1=le.fit_transform(labels1)
print(labels1[:10])

# tokenized_texts1 = [tokenizer.tokenize(sent) for sent in sentences1]

# MAX_LEN = 128
# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary

input_ids1=[inputs["input_ids"] for inputs in sentences_test]
# Pad our input tokens
input_ids1 = pad_sequences(input_ids1, maxlen=MAX_LEN, truncating="post", padding="post")
# if inter==1:
token_type_ids1=[inputs["token_type_ids"] for inputs in sentences_test]
token_type_ids1=pad_sequences(token_type_ids1, maxlen=MAX_LEN,truncating="post", padding="post")
# Create attention masks
attention_masks1 = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids1:
  seq_mask1 = [float(i>0) for i in seq]
  attention_masks1.append(seq_mask1)





prediction_inputs = torch.tensor(input_ids1).to(torch.int64)
prediction_masks = torch.tensor(attention_masks1).to(torch.int64)
# if inter==1:
prediction_types = torch.tensor(token_type_ids1).to(torch.int64)
prediction_labels = torch.tensor(labels1).to(torch.int64)


batch_size = 32 

# if inter==1:
prediction_data = TensorDataset(prediction_inputs, prediction_types,prediction_masks, prediction_labels)

# elif intra==1:
# prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)



# Prediction on test set

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  # Unpack the inputs from our dataloader
  # if inter==1:
  b_input_ids,b_type, b_input_mask, b_labels = batch
  # # Telling the model not to compute or store gradients, saving memory and speeding up prediction
  with torch.no_grad():
    # Forward pass, calculate logit predictions
    logits = model(b_input_ids, token_type_ids=b_type, attention_mask=b_input_mask)

  # b_input_ids, b_input_mask, b_labels = batch
  # Telling the model not to compute or store gradients, saving memory and speeding up prediction
  # with torch.no_grad():
  #   # Forward pass, calculate logit predictions
  #   logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print(len(predictions),len(true_labels))
print(predictions[0][0])

# Import and evaluate each test batch using Matthew's correlation coefficient
from sklearn.metrics import matthews_corrcoef
matthews_set = []

for i in range(len(true_labels)):
  matthews = matthews_corrcoef(true_labels[i],
                 np.argmax(predictions[i], axis=1).flatten())
  matthews_set.append(matthews)



flat_true_labels1=[]
# Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]
# flat_true_labels1.extend([item for sublist in true_labels for item in sublist])
# print(len(tags_vals1),len(flat_predictions),len(flat_true_labels))

# pred_tags = [tags_maps[p] for p in flat_predictions]
# pred_tags[:10]

print(len(flat_predictions),len(flat_true_labels))

# print(flat_predictions[989:994],flat_true_labels[989:994])
print(flat_predictions[0:11],flat_true_labels[0:11])

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print('Classification Report')
labels_test=flat_predictions
# pred = model.predict(xte)
# c = np.argmax(pred, axis=-1)
print(classification_report(flat_true_labels,flat_predictions))
print(confusion_matrix(flat_true_labels,flat_predictions))

