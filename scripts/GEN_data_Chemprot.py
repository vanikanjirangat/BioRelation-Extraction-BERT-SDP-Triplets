# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:09:07 2019

@author: Install
"""
import csv
import nltk
import regex
from pattern.en import parsetree
from pandas import DataFrame
import itertools
train_rel='.\\craft-st-master\\ChemProt_Corpus\chemprot_training\chemprot_training\chemprot_training_relations.tsv'
train_abstracts='.\\craft-st-master\\ChemProt_Corpus\chemprot_training\chemprot_training\chemprot_training_abstracts.tsv'
train_entities='.\\craft-st-master\\ChemProt_Corpus\chemprot_training\chemprot_training\chemprot_training_entities.tsv'

def _extract(train_abstracts,train_entities,train_rel):
    pmids=[]; texts=[]
    
    with open(train_abstracts,'r', encoding='utf-8') as f:
        f1=csv.reader(f,delimiter='\t')
        for record in f1:
    #        print(record[0],record[1],record[2])
            pmid=record[0]
            text=[record[1]]
            text.append(record[2])
            text1=text[0]+' '+text[1]
            pmids.append(pmid)
            texts.append(text1)
            
    
    entity_map={}        
    with open(train_entities,'r', encoding='utf-8') as f:
        f2=csv.reader(f,delimiter='\t')
        for record in f2:
            #11319232	T1	CHEMICAL	242	251	acyl-CoAs
            if record[0] not in entity_map.keys():
                entity_map[record[0]]=[(record[1:])]
            else:
                entity_map[record[0]].append((record[1:]))
            
    #FOR TESTING ACTUALLY THIS WILL NOT BE PRESENT!!!!        
    relation_id_map={}
    relations_expanded={}
    relations_args={}
    with open(train_rel,'r', encoding='utf-8') as f:
        f2=csv.reader(f,delimiter='\t')
        for record in f2:
            #16437532	CPR:4	Y 	INHIBITOR	Arg1:T11	Arg2:T39
            #label-info=record[3]
            if record[0] not in relation_id_map.keys(): 
                if record[2]  =='Y ':
                    relation_id_map[record[0]]=[(record[1:])]
            else:
                if record[2]  =='Y ':
                    relation_id_map[record[0]].append((record[1:]))  
            
            
            entities=entity_map[record[0]]# get the corresponding entities
            entity={}
            for i in entities:
                entity[i[0]]=i[4]
                
            if record[0] not in relations_expanded.keys():
                if record[2]  =='Y ':
                    relations_expanded[record[0]]=[(record[1],record[2],record[3],entity[record[4].split(':')[1]],entity[record[5].split(':')[1]])]
            else:
                if record[2]  =='Y ':
                    relations_expanded[record[0]].append((record[1],record[2],record[3],entity[record[4].split(':')[1]],entity[record[5].split(':')[1]]))
            if record[0] not in relations_args.keys():
                if record[2]  =='Y ':
                    relations_args[record[0]]=[(record[1],record[2],record[3],record[4].split(':')[1],record[5].split(':')[1])]
            else:
                if record[2]  =='Y ':
                    relations_args[record[0]].append((record[1],record[2],record[3],record[4].split(':')[1],record[5].split(':')[1])) 
    return pmids,texts,entity_map,relation_id_map, relations_expanded,relations_args

import re

def _create_dataset(pmids,texts,entity_map,relation_id_map, relations_expanded,relations_args):
    pos=[]
    neg=[]
#    pm='10047461'
    
    
    for _,pmid in enumerate(pmids):
#        if pmid==pm:
        ent=entity_map[pmid]
        ent=[x for x in ent if x[1]!='GENE-N'] 
        chemical_list=[k[-1] for k in ent if k[1]=='CHEMICAL']
        chemical_list=list(set(chemical_list))
        gene_list=[k[-1] for k in ent if k[1]=='GENE-Y']
        gene_list=list(set(gene_list))
        entity_pairs=list(itertools.product(chemical_list,gene_list))
        entity_pairs=list(set(entity_pairs))
        texts1=texts
        ab=texts1[_]  
        ab1=nltk.sent_tokenize(ab) 
#        print('ent:',ent)
        pp=[]
        nn=[]
        
        
        if pmid in relations_expanded.keys():
            entity_rel=relations_expanded[pmid]
   
            positive_pairs=[(k[3],k[4]) for k in entity_rel]
            negative_pairs=[x for x in entity_pairs if x not in positive_pairs]
#                print(negative_pairs)
            
            for j,items in enumerate(positive_pairs):
                for i,sent in enumerate(ab1):
                    if items[0] in sent and items[1] in sent:
                        label=entity_rel[j][0]
                        pp.append((sent,items,label))
            
            for j,items in enumerate(negative_pairs):
                for i,sent in enumerate(ab1):
                    if items[0] in sent and items[1] in sent:
                        label='NOREL'
                        nn.append((sent,items,label))    
            
        else:
            
            for j,items in enumerate(entity_pairs):
                for i,sent in enumerate(ab1):
                    if items[0] in sent and items[1] in sent:
                        label='NOREL'
                        nn.append((sent,items,label))  
        if len(pp)>=1:
            pos.append(pp)
        if len(nn)>=1:
            neg.append(nn)
                        
                    

    return pos,neg

pmids,texts,entity_map,relation_id_map, relations_expanded,relations_args=_extract(train_abstracts,train_entities,train_rel)
pos,neg= _create_dataset(pmids,texts,entity_map,relation_id_map, relations_expanded,relations_args)   

pos1=[]
neg1=[]
samples=[]
for k in pos:
    pos1.extend(k)
for k1 in neg:
    neg1.extend(k1)
samples.extend(pos1)
samples.extend(neg1)
#phrasal_positive=pos1
#phrasal_negative=neg1
#print('positive and negative instances',len(phrasal_positive),len(phrasal_negative))        
##positive examples with a label
#phrasal_pos=[k[0] for k in phrasal_positive]
#labels_pos=[k[1] for k in phrasal_positive]
##labels_def_pos=[k[2] for k in phrasal_positive]
##negative examples with a label
#phrasal_neg=[k[0] for k in phrasal_negative]
#labels_neg=[k[1] for k in phrasal_negative]
##labels_def_neg=[k[2] for k in phrasal_negative[:3500]]
#
#phrases=[*phrasal_pos, *phrasal_neg]
#
#labels=[*labels_pos, *labels_neg]
#print('positive and negative instances',len(phrases),len(labels)) 
##CREATING THE TRAINING SAMPLE(NEGATIVE AND POSITIVE EXAMPLES)
#df = DataFrame({'Sent':phrases ,'class_type':labels })       
#df.to_excel('C:\\Users\\Install\\Miniconda3\\envs\\tensorflow_env\\OGER\\PYTORCH_biobert_tuning\\CHEMPROT_DATA\\CHEMPROT_TRAIN_FULL.xlsx', sheet_name='sheet1', index=False)     
#
#
pred_dir='C:\\Users\\Install\\Miniconda3\\envs\\tensorflow_env\\OGER\\PYTORCH_biobert_tuning\\CHEMPROT_DATA\\PAIR_DATA\\Train1.tsv'
with open(pred_dir, 'w',encoding='utf-8') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t',lineterminator='\n')
    for _,r1 in enumerate(samples): 
        if len(r1)>0:
    #                        writer.writerow([r1[0], r1[-1].split(':')[0]])
            writer.writerow([r1[0],r1[1],r1[2]])
        else:
            writer.writerow(r1)


###TEST SET CONSTRUCTION#####
test_abstracts='.\\craft-st-master\\ChemProt_Corpus\chemprot_test_gs\chemprot_test_gs\chemprot_test_abstracts_gs.tsv'
test_entities='.\\craft-st-master\\ChemProt_Corpus\chemprot_test_gs\chemprot_test_gs\chemprot_test_entities_gs.tsv'
test_relations='.\\craft-st-master\\ChemProt_Corpus\chemprot_test_gs\chemprot_test_gs\chemprot_test_relations_gs.tsv'

pmids1,texts1,entity_map1,relation_id_map1, relations_expanded1,relations_args1=_extract(test_abstracts,test_entities,test_relations)
#
pos11,neg11= _create_dataset(pmids1,texts1,entity_map1,relation_id_map1, relations_expanded1,relations_args1) 
pos2=[]
neg2=[]
samples_t=[]
for k in pos11:
    pos2.extend(k)
for k1 in neg11:
    neg2.extend(k1)
samples_t.extend(pos2)

samples_t.extend(neg2)
#
pred_dir='C:\\Users\\Install\\Miniconda3\\envs\\tensorflow_env\\OGER\\PYTORCH_biobert_tuning\\CHEMPROT_DATA\\PAIR_DATA\\Test1.tsv'
with open(pred_dir, 'w',encoding='utf-8') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t',lineterminator='\n')
    for _,r1 in enumerate(samples_t): 
        if len(r1)>0:
    #                        writer.writerow([r1[0], r1[-1].split(':')[0]])
            writer.writerow([r1[0],r1[1],r1[2]])
        else:
            writer.writerow(r1)
            
