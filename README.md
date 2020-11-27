# Enhancing Biomedical Relation Extraction with Transformer Models using Shortest Dependency Path Features and Triplet Information

In the paper, we propose utilizing shortest dependency path (SDP) features for constructing the data samples by pruning out the noisy information and selecting the most representative samples for model learning. We also utilize triplet information in model learning with the biomedical variant of BERT, viz., BioBERT, by representing the problem as a sentence pair classification task and using the (sentence, triplet information) pair. propose to utilize SDP features for data sample slections and effective noisy sample pruning. Further, we utilize the triplet information for model learning.

## Data Generation: 
*Gen_data_CDR.py*-: script to generate the Train, Dev and Test sets for CDR dataset using SDP information  
The script in turn utilizes:  
*scispacy_parse_intra.py* and *scispacy_parse_pair.py*  
*scispacy_parse_intra.py*-: SDP comptation for intra-sentential relations  
*scispacy_parsepair.py*-:SDP comptation for inter-sentential relations  
*Gen_data_random_CDR*-:script to generate the train, dev and test sets for CDR dataset without SDP  

## Steps 

First generate the samples using the scripts specified in *Data Generation*. Once the *Train*, *Dev* and *Test* sets are constructed, we have to Train the model.


### Model Training & Testing

BioBERT model is used. You have to download the pretrained model files for BioBert from *https://github.com/dmis-lab/biobert* and convert the checkpoints using the instructions in *https://huggingface.co/transformers/converting_tensorflow_models.html* and store them in the required path. 

*CDR_biorel.py*: This script utilizes the data generated by *Gen_data_CDR.py* and and fine tune the BioBERT model uisng *Train* & *DEV* sets and make predictions on *Test set*.  
The paths of input directory, the model path and output directory must be set properly. 




## Requirements
torch==1.1.0

spacy==2.3.2 *(download the model pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_sm-0.3.0.tar.gz)*

tensorflow==1.12.0

Keras==2.2.4

scispacy==0.3.0

matplotlib==3.0.2

transformers==2.11.0

tqdm==4.32.2

nltk==3.4

pandas==0.24.2

numpy==1.17.0

scikit_learn==0.23.2


