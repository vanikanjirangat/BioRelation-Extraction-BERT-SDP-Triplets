# BioRelation_Extraction_BioBERT_SDP
Biomedical Relation Extraction by enhancing the tranformer models reducing the noisy samples utilizing shortes dependency features (SDP).

# STEPS



# Data Generation: Removing Nosiy Data with SDP Information
Gen_data_CDR.py-- script to generate the train, dev and test sets for CDR dataset
Gen_data_Chemprot.py--script to generate the train, dev and test sets for Chemprot dataset


These scripts in turn utilizes the scispacy_parse_intra.py and scispacy_parse_pair.py.
scispacy_parse_intra.py-- SDP comptation for intra-sentential relations
scispacy_parse_pair.py--SDP comptation for inter-sentential relations

# Data Generation: WIthout SDP Information
Gen_data_random_CDR--script to generate the train, dev and test sets for CDR dataset without SDP

# Modelling 

cdr_bio_pair_sdp: The script utilizes the data generated by Gen_data_CDR.py and BioBERT model is used.
The paths of input directory, the model path and output directory must be set properly. Chemprot evaluations can be done similarily just changing the # of labels  accordingly. For Chemprot (#  of labels=6). 
You have to download the pretrained model files for BioBert from https://github.com/dmis-lab/biobert and convert the checkpoints using pytorch_weights.py script

# Dependencies
torch==1.1.0

spacy==2.2.3

tensorflow==1.12.0

Keras==2.2.4

scispacy==0.2.4

networkx==2.3

matplotlib==3.0.2

pytorch_pretrained_bert==0.6.2

tqdm==4.32.2

nltk==3.4

transformers==2.11.0

pandas==0.24.2

numpy==1.17.0

ipython==7.17.0

scikit_learn==0.23.2
