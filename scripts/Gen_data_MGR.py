# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:36:07 2020

@author: Install
"""

'''
Extract information from CDR data: for CID subtask
On sample data
'''
import os
import argparse
import nltk
import itertools
import csv
from collections import OrderedDict
from scispacy_parse_pair import get_parsed
from scispacy_parse_intra import sdp
import re
class preprocess_model(object):
    def __init__(self, args):
        self.args = args
        self.opath = args.opath
        self.bioc = args.biocpath
        self.pubtator = args.pubtator
        if not os.path.isdir(self.opath):
            os.makedirs(self.opath)
            
            
    def extract_data(self,name):
        file =self.pubtator+name
        self.data_dict={}
        with open(file,encoding='utf-8') as f:
            doc=f.read()
        doc=doc.split('\n\n')
        docs=[x.split('\n') for x in doc]
        
        info=[[x[0],x[1],x[2:]] for x in docs if len(x)>2]
        
        self.info_ext=[[x[0],x[1],[y.split('\t') for y in x[2]]] for x in info]
        
#        ids=d[0].split('|')[0]
        return self.info_ext
    
  
            
            
            
    def extract_rel_data(self):
        data=self.info_ext
#        data=[data[2]]
        self.relations=[]
        self.relation_list=[]
        self.count_entity=0.0
        self.count_p1=0.0
        self.count_p=0.0
        self.count_n1=0.0
        self.id_chem=[]
        self.id_dis=[]
        self.pairs=[]
        self.p_rel=0
        self.positive_pairs=[]
        self.pmid_rel={}
        self.pmid_ct1={}
        self.relations_train={}
        self.relations_test={}
        
        with open('./data/PMID_TestSet.txt') as test_ids:
            test_ids=test_ids.read()
        test_ids=test_ids.split('\n')
        test_ids=[x for x in test_ids if len(x)>0]
        for d in data:
            self.ent={}
            train=0
            test=0
            relations1=[]
            cids=[]
            p=[]
            n=[]
            self.entity=[]
            self.entity_info=[]
            
            pmid=d[0].split('|')[0]
            
            if pmid not in test_ids:# extracting training set
                train=1
            else:
                test=1# else test
            title=d[0].split('|')[2]
            
            abstract=d[1].split('|')[2]
            
            tags=d[2]
#            print(tags)
           
#            if pmid=='11774264':
            text=title+' '+abstract
#                print(text)
            
            
#                print('#####################yes')
#                print(tags)
            rel=[k for k in tags if k[1]=='GID']
#            print('*****rel*****',rel)
            
            '''
            Consider the rels like [['17085653', 'GID', '25855', 'D008545']]
            '''
            r_pairs=[]
#            print(rel)
            for r in rel:
                if len(r)==5:
                    r_pairs.extend(r[-1].split(' '))
#                else:
#                    r_pairs.extend([r[-2],r[-1]])
            
             #geneid_offsetstart_offsetend   
#            print(r_pairs)
            r_pairs=[x.split(',') for x in r_pairs]
            r_pairs=[(x[0].split('_'),x[1].split('_')) for x in r_pairs]
#                print(r_pairs)
                
            tgs=[k[1:] for k in tags]
#                print(tgs)
            c1=[k for k in tgs if k[1]!='GID' and k[2]!='-1' and k[-2]=='Gene']
            d1=[k for k in tgs if k[1]!='GID' and k[2]!='-1' and k[-2]=='Disease']
            c2=[[x[-1],x[0],x[1]] for x in c1]
            d2=[[x[-1],x[0],x[1]] for x in d1]
#                print(c1)
#                print(d1)
            entity_ids=list(itertools.product(c2,d2))
            
            n_pairs=[x for x in entity_ids if x not in r_pairs]
            p_pairs=[x for x in entity_ids if x in r_pairs]
            pids=[]
            for x in p_pairs:
                pids.append((x[0][0],x[1][0]))
            
            pids=list(set(pids))
#            print('****',pids)
#            print(len(n_pairs),len(p_pairs),len(r_pairs),len(entity_ids))
#                print(entity_ids)
            '''
            rearrange and compare. So those in the r_pairs are positive and remaining negative
            Then see whether we have both intra and inter-level sentences to tackle. Now, for the same entity
            pair, we can have multiple sentence occurrences?
            Trying with sentence pair classification
            HERE the OFFSET of relations are also given
            '''
            offsets_sentences=[]
            for match in re.finditer(r'(?s)((?:[^.]?)+)', text):
                if match.start()!=match.end():
                    offsets_sentences.append((match.start(), match.end()))
#                print(offsets_sentences)
#                print(p_pairs)
            intras_p=[]
            inters_p=[]
            rels=[]
            entity_rel={}
            for p1 in p_pairs:
                
                s1=''
                s2=''
                for off in offsets_sentences:
                    if len(s1)==0:
                        s1=''
                    if len(s2)==0:
                        s2=''
                    
                    start=int(p1[0][1])
                    end=int(p1[0][2])
                    e1=text[start:end]
                    if start>=off[0] and end<=off[1]:
                        s1=text[off[0]:off[1]]
#                            print(s1)
                    
                    start=int(p1[1][1])
                    end=int(p1[1][2])
                    e2=text[start:end]
                    if start>=off[0] and end<=off[1]:
                        s2=text[off[0]:off[1]]
#                            print(s2)
                if s1!='' and s2!='':
                    if s1==s2:
                        intras_p.append((s1,e1,e2,1,'intra'))
                    else:
                        inters_p.append((s1,s2,e1,e2,1,'inter'))
                
#                print(intras_p)
#                print(inters_p)
            if len(intras_p)>0:
                intras_p=list(set(intras_p))
                rels.extend(intras_p)
            else:
                if len(inters_p)>0:
                    inters_p=list(set(inters_p))
                    rels.extend(inters_p)
                   
#            print(len(intras_p),len(inters_p))
            
            intras_n=[]
            inters_n=[]
            '''
            TODO : mark the positions 
            '''
            for p1 in n_pairs:
                nid=(p1[0][0],p1[1][0])
                if nid not in pids:

                    s1=''
                    s2=''
                    for off in offsets_sentences:
                        if len(s1)==0:
                            s1=''
                        if len(s2)==0:
                            s2=''
                        
                        start=int(p1[0][1])
                        end=int(p1[0][2])
                        e1=text[start:end]
                       
                        if start>=off[0] and end<=off[1]:
#                            print(off)
                            s1=text[off[0]:off[1]]
                            
                        
                        start=int(p1[1][1])
                        end=int(p1[1][2])
                        e2=text[start:end]
                        
                        if start>=off[0] and end<=off[1]:
#                            print(off)
                            s2=text[off[0]:off[1]]
                            
                    
                    if s1!='' and s2!='':
                        
                        if s1==s2:
                            intras_n.append((s1,e1,e2,0,'intra'))
#                            print(intras_n)
                        else:
                            inters_n.append((s1,s2,e1,e2,0,'inter'))
#                            print(inters_n)
            if len(intras_n)>0:
                intras_n=list(set(intras_n))
                rels.extend(intras_n)
            else:
                if len(inters_n)>0:
                    inters_n=list(set(inters_n))
                    rels.extend(inters_n)
#            print(len(intras_n),len(inters_n))
#            print(rels[0])
                    
            '''
            correct the index
            (' ALA-SDT showed synergistic anti-tumor effects in malignant melanoma by constituting a positive feedback loop of p53-miR-34a-Sirt1 axis', 'miR-34a', 'malignant melanoma', 1, 'intra')
            '''
            
            for item in rels:
                epair=(item[-4],item[-3])
                if rels[-1]=='intra':
                    if epair not in self.ent.keys():
                        
                        self.ent[epair]=[(item[0],item[-2],item[-1])]
                        
                    else:
                        self.ent[epair].append((item[0],item[-2],item[-1]))
                else:
                    if epair not in self.ent.keys():
                        
                        self.ent[epair]=[(item[0],item[1],item[-2],item[-1])]
                        
                    else:
                        self.ent[epair].append((item[0],item[1],item[-2],item[-1]))
                        
            self.ent_nw={}
            for items in self.ent.keys():
#                if len(self.ent[items])>1:
                pl_intra=[]
                pw_intra=[]
                triplet_intra=[]
                pl_inter=[]
                pw_inter=[]
                triplet_inter=[]
#                print('sdp###')
                for samp in self.ent[items]:
                    if samp[-1]=='intra':
                        label=samp[-2]
                        pair=items
                        pl,pw,d,e1,e2,r=sdp(samp[0],pair) 
                    
                        pl_intra.append(pl)
                        pw_intra.append(pw)
                        triplet_intra.append([e1,r,e2])
                    
                    else:
                        pair=items
#                        if label==0:
                        s1=samp[0]
                        s2=samp[1]
                        label=samp[-2]
#                        print(s1,pair[0])
                        pl1,pw1,d1,e1,e11=get_parsed(s1,pair[0]) 
                        pl2,pw2,d2,e2,e22=get_parsed(s2,pair[1]) 
                        pw2.reverse()
                        pw=pw1+pw2
                        pw_inter.append(pw)
                        triplet_inter.append([e1,e11,e22,e2])
                        path=pw[1:len(pw)-1]
                        pl=len(path)
                        pl_inter.append(pl)
#                        else:
#                             sampled.append(item)
#                             s=item[0]+' '+item[1]
#                             self.samples.append((s,pair,label))
                    
                    
                        
                    
                if len(pl_intra)>0:       
                    idx=pl_intra.index(min(pl_intra))
                    sel_intra=self.ent[items][idx]
                    
                    s_path=pw_intra[idx]
                    triplet=triplet_intra[idx]
                    sent=sel_intra[0]
                    sent=sent.replace(items[0],'['+items[0]+']')
                    sent=sent.replace(items[1],'['+items[1]+']')
                    if items not in self.ent_nw.keys():
                        self.ent_nw[items]=[(sent,triplet,s_path,label,'intra')]
                if len(pl_inter)>0:
                    idx=pl_inter.index(min(pl_inter))
                    sel_inter=self.ent[items][idx]
                    s_path=pw_inter[idx]
                    s_triplet=triplet_inter[idx]
                    sent1=sel_inter[0][0]
                    sent1=sent1.replace(items[0],'['+items[0]+']')
                    sent2=sel_inter[1][0]
                    sent2=sent2.replace(items[1],'['+items[1]+']')
                    sel_inter=sent1+','+sent2
                    sel_inter=sel_inter[0][0]+' '+sel_inter[1][0]
#                        sel=sel_inter[0]+sel_inter[2]
                    if items not in self.ent_nw.keys():
                        self.ent_nw[items]=[(sel_inter,s_triplet,s_path,'inter')]
                
                            
                         
                        
                   
                
               
                    
                    
            '''
            If no. of senteces corrsponding to an entity pair is more than 1,
            select uisng SDP information. First priority to intra
            '''
#            print('ent_list',len(list(self.ent.items())))
            if train:
                if pmid in self.relations_train.keys():
                    self.relations_train[pmid].append(self.ent_nw)
                else:
                    self.relations_train[pmid]=self.ent_nw
            
            if test:
                if pmid in self.relations_test.keys():
                    self.relations_test[pmid].append(self.ent_nw)
                else:
                    self.relations_test[pmid]=self.ent_nw
#            print(self.relations)
        
        return self.relations_train,self.relations_test
                    

#
    
    
#    Creating positive and negative samples from the text. The relation will be binary either 0/1
    
    def write_path(self):
        
#        pred_dir=os.path.join(self.opath,'sample_cdr_aux2.tsv')
#        under=os.path.join(self.opath,'under_sampled_aux2.tsv')
#        multi=os.path.join(self.opath,'multi_sampled_aux2.tsv')
##        full=os.path.join(self.opath,'full_sampled_bal_aux.tsv')
        test=os.path.join(self.opath,'test1.tsv')
        train=os.path.join(self.opath,'train1.tsv')
        with open(train, 'w',encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t',lineterminator='\n')
            for rr in self.relations_train.keys(): 
                for r in self.relations_train[rr].keys():
    #                print(r1)
#                    print(r)
                    if len(self.relations_train[rr][r])>0:
                        r1=self.relations_train[rr][r][0]
#                        print(r1)
                        if len(r1[0])>3:
#                            print(r1)
                            writer.writerow([r1[0],r,r1[1],r1[2],r1[3],r1[4]])
        with open(test, 'w',encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t',lineterminator='\n')
            
            for rr in self.relations_test.keys(): 
                for r in self.relations_test[rr].keys():
    #                print(r1)
#                    print(r)
#                    print(r,rr,self.relations_test[rr])
                    if len(self.relations_test[rr][r])>0:
                        r1=self.relations_test[rr][r][0]
#                        print(r1)
                        if len(r1[0])>3:
#                            print(r1)
                            writer.writerow([r1[0],r,r1[1],r1[2],r1[3],r1[4]])
 
        
        
def parse_args():
        parser = argparse.ArgumentParser(description="Parse the data")

        
#        parser.add_argument('--biocpath', type=str, default='data/',
#                            help='Folder name training.')
        
        parser.add_argument('--pubtator', type=str, default='data/',
                            help='Folder name pubtator data.')

        

        parser.add_argument('--opath', type=str, default='datasets/',
                            help='Name of the desired output folder.')

        

        args =  parser.parse_args()
        return args
    
args = parse_args()
#
## Train TWEC model
##args=['--train .\\train\\']
m = preprocess_model(args)
#
data=m.extract_data('MGR_dataset_v0.5.txt')
##CDR_DevelopmentSet.PubTator
##CDR_TestSet.PubTator
##CDR_TrainingSet.PubTator
##CDR_sample.PubTator.txt
train,test=m.extract_rel_data()

#
m.write_path()


