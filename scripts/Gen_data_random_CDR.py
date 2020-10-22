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
import random

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
        with open(file) as f:
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
        for d in data:
            relations1=[]
            cids=[]
            p=[]
            n=[]
            self.entity=[]
            self.entity_info=[]
            
            title=d[0].split('|')[2]
            abstract=d[1].split('|')[2]
            pmid=d[0].split('|')[0]
#            print(pmid)
            tags=d[2]

            tgs=[k[3:] for k in tags]
            c1=[k for k in tgs if len(k)>=3 and k[2]!='-1' and k[1]=='Chemical']
            d1=[k for k in tgs if len(k)>=3 and k[2]!='-1' and k[1]=='Disease']
            
            
            
#                print(d1)
            chemical_list=[]
            for  k in c1:
                if k[-1]=='':
                    k.remove(k[-1])
                if len(k)==4:
                    
#                    print(k)
                    ids=k[-2].split('|')
                    
#                    if len(ids)>1:
                        
                    m1=[k[0],k[1],ids[0]]
                    m2=[k[0],k[1],ids[1]]
                    if m1 not in chemical_list:
                        chemical_list.append(m1)
                    if m2 not in chemical_list:
                        chemical_list.append(m2)
                else:
                    if k not in chemical_list:
                        chemical_list.append(k)
                    
            
            disease_list=[]
            
            for  k in d1:
                if k[-1]=='':
                    k.remove(k[-1])
                if len(k)==4:
                    
                    
                    ids=k[-2].split('|')
                    m1=[k[0],k[1],ids[0]]
                    m2=[k[0],k[1],ids[1]]
                    if m1 not in disease_list:
                        disease_list.append(m1)
                    if m2 not in disease_list:
                        disease_list.append(m2)
                else:
                    if k not in disease_list:
                        disease_list.append(k)
            chemical_list=list(map(list, OrderedDict.fromkeys(map(tuple, chemical_list))))
        
            disease_list=list(map(list, OrderedDict.fromkeys(map(tuple, disease_list))))    
#                print(chemical_list)
            
            chem_ids=[x[-1] for x in chemical_list]
            dis_ids=[x[-1] for x in disease_list]
            entity_ids=list(itertools.product(chem_ids,dis_ids))
            
            
            '''
            [pmid,'cid','D007213', 'D007022']
            '''
            cids=[k for k in tags if len(k)==4]
#                print('cids',cids)
            
            rels=[(k[-2],k[-1]) for k in cids]
            
            

            
            self.pmid_ct1[pmid]=len(rels)
                    
            for r in rels:
                if r in entity_ids:
                    self.p_rel+=1
            
            

            
            #GROUPING AS UNIQUE
            chem_list={}
            for ids in chem_ids:
                for item in chemical_list:
                    if ids in item:
#                        if ids!=-1:
                        if ids not in chem_list.keys():
                            chem_list[ids]=[item[0]]
                        else:
                            chem_list[ids].append(item[0])
            
            dis_list={}
            for ids in dis_ids:
                for item in disease_list:
                    if ids in item:
#                        if ids!=-1:
                        if ids not in dis_list.keys():
                            dis_list[ids]=[item[0]]
                        else:
                            dis_list[ids].append(item[0])
                            

            chem1=[]
            dis1=[]
            
            #REPLACING WITH THE STRING OF MINIMUM CHARACTERS
            for chem in chem_list.keys():
                g=[x for x in chem_list[chem]]
                g1=[len(x) for x in chem_list[chem] ]
                g2=g1.index(min(g1))
                rep=g[g2]
                c=[rep,chem]
                chem1.append(c)
#                print(g,rep)
                for k in chem_list[chem]:
                    if k in title or k in abstract:
                        
                        title=title.replace(k+' ',rep+' ')
                        abstract=abstract.replace(k+' ',rep+' ')
                
            for dis in dis_list.keys():
                g=[x for x in dis_list[dis]]
                g1=[len(x) for x in dis_list[dis] ]
                g2=g1.index(min(g1))
                rep=g[g2]
                d=[rep,dis]
                
                dis1.append(d)
                for k in dis_list[dis]:
                    if k in title or k in abstract:
#                        
                        ## replace only exact match
                        title=title.replace(k+' ',rep+' ')
                        abstract=abstract.replace(k+' ',rep+' ')
            
            
            self.id_chem.append(chem_list)
            self.id_dis.append(dis_list)

            entity_pairs=list(itertools.product(chem1,dis1))
            
#            print(pmid)
            assert len(entity_pairs)==len(chem1)*len(dis1)

            self.pairs.extend(entity_pairs)
            
            title1=title
            abstract1=abstract
            
#                print(disease_list)
            for chem in chemical_list:
                if chem[0] in title1:
                    
                    title1=title1.replace(chem[0],chem[-1])
                if chem[0] in abstract1:
                    abstract1=abstract1.replace(chem[0],chem[-1])
            
            for dis in disease_list:
                if dis[0] in title1: 
                    
                    title1=title1.replace(dis[0],dis[-1])
                if dis[0] in abstract1:
                    abstract1=abstract1.replace(dis[0],dis[-1])
            

            #the no.of positive relations
            pair_cids=[(x[-2],x[-1]) for x in cids]
            
            pair_cids1=[(x[-1],x[-2]) for x in cids]
            
            entity_id_pair=[(pair[0][-1],pair[1][-1]) for pair in entity_pairs]
            
            entity_name_pair=[(pair[0][0],pair[1][0]) for pair in entity_pairs]
            
            ct=0
            for i,pair in enumerate(pair_cids):
                if pair in entity_id_pair:
                    ct+=1
#                    self.count_p1+=1
                elif pair_cids1[i] in entity_id_pair:
                    ct+=1

            self.count_p+=len(cids)
            

            for pair in entity_pairs:
                #an (chem,disease) entity pair in a pmid
               
                flag=0
                for rid in cids:
                    # to get the true label, we label 1 if the relation in cid
#                    
                    if pair[0][-1] in rid and pair[1][-1] in rid:
#                            print(pair)
                        
                        flag=1
#                        self.p_rel+=1
                        
#                        print('%%p',(pair[0][-1],pair[1][-1]))
                        #positive relations
                if flag==1:
#                        print((pair[0][0],pair[1][0]))
                    '''
                    POSITIVE PAIRS
                    '''
                    self.count_p1+=1
                    p.append((title,abstract,(pair[0][0],pair[1][0])))
#                        print(p)
#                    print('%%p',(pair[0][-1],pair[1][-1]))
    #                        self.count_p+=1
                    self.relations.append((title,abstract,(pair[0][0],pair[1][0]),1))
                    
                    relations1.append((title,abstract,(pair[0][0],pair[1][0]),1))
                    self.positive_pairs.append((pair[0][0],pair[1][0]))
                    
                else:
                    '''
                    NEGATIVE PAIRS
                    '''
#                    self.count_n1+=1
#                    print('%%n',(pair[0][-1],pair[1][-1]))
                    self.relations.append((title,abstract,(pair[0][0],pair[1][0]),0))
                    relations1.append((title,abstract,(pair[0][0],pair[1][0]),0))
                    n.append((title,abstract,(pair[0][0],pair[1][0])))
                    
            p=list(set(p))
            n=list(set(n))
            

#                relations1.append((t[0],t[1],t[2],0))
            self.relations=list(set(self.relations))
            relations1=list(set(relations1))
#            self.count_p1+=len(relations1)
            self.relation_list.append((relations1,pmid,entity_pairs))
    #            for t in negs:
    #                self.relations.append(negs,0))
                
    #        print(self.relations[0])
    #        print(self.count_entity,self.count_p,self.count_n,len(self.relations))
    #        print('|||',len(relations1))
#            print(len(self.relation_list))
        return self.relations,self.relation_list
    
    
    def formulate_samples(self):
        self.samples=[]
        
        self.full_sample=[]
        self.pp=0
        self.nn=0
        self.t=0
        
        self.positive_trace=0
        self.negative_trace=0
        ##FROM THE ENTITY PAIRS WE CREATE THE SAMPLES
        #we need to extract both inter and intra sentence relations
        #for a pmid
        for pmval in self.relation_list:
            pmid=pmval[1]

            
            for items in pmval[0]:
#                print(len(items))
                
                
                s=[]
                
                title,abstract,pair,label=items[0],items[1],items[2],items[3]
                
                
                full_abstract=title+' '+abstract
                
                
                
#                print(full_abstract)
                
                
#                print('\n\n\n\n')
                
                sent_text=nltk.sent_tokenize(full_abstract)
                
                      
                  
               
                
                
                
                '''
                GET THE INTRA-LEVEL & INTER-LEVEL SENTENCES
                '''
                
                intra=[]
                
                
                p1=[]
                p2=[]
                ff=0
                inter=[]
                for i,sent in enumerate(sent_text):
                    if '-induced' in sent:
                        sent=sent.replace('-induced',' induced')
                    if '-associated' in sent:
                        sent=sent.replace('-associated',' associated')
                    
                    if (pair[0] in sent and pair[1] in sent) or (pair[0] in sent.lower() and pair[1] in sent.lower()):
                        sent1=sent
                        pr=('['+pair[0]+']chemical','['+pair[1]+']disease')
                        sent=sent.replace(pair[0],'['+pair[0]+']chemical',1)
                        sent=sent.replace(pair[1],'['+pair[1]+']disease',1)
                        intra.append((sent,sent1))
                        
                        ff=1
                    
                if ff==0:
                    for i,sent in enumerate(sent_text):
                        if '-induced' in sent:
                            sent=sent.replace('-induced',' induced')
                        if '-associated' in sent:
                            sent=sent.replace('-associated',' associated')
                        if (pair[0] in sent) or (pair[0] in sent.lower()):
                            sent1=sent
                            sent=sent.replace(pair[0],'['+pair[0]+']chemical',1)
                            p1.append((sent,sent1))
                        if (pair[1] in sent) or (pair[1] in sent.lower()):
                            sent1=sent
                            sent=sent.replace(pair[1],'['+pair[1]+']disease',1)
                            p2.append((sent,sent1))
                                
                            
                inter=list(itertools.product(p1,p2))
                inter=list(set(inter))

                if len(intra)>=1:
                    if label==1:
                        
                        self.positive_trace+=1
                    else:
                        self.negative_trace+=1
                if len(inter)>=1:
                    if label==1:
                        self.positive_trace+=1
                    else:
                        self.negative_trace+=1
                
                sampled=[]
                
                pr=('['+pair[0]+']chemical','['+pair[1]+']disease')
                if len(intra)>1:
                    
                    pl_intra=[]
                    pw_intra=[]
                    triplet_intra=[]
                    #considering path lenghts for all samples
                    for item in intra:
                       
                        pl,pw,d,e1,e2,r=sdp(item[-1],pair) 
                        
                        pl_intra.append(pl)
                        pw_intra.append(pw)
                        triplet_intra.append([e1,r,e2])
                        
                        
                    if len(pl_intra)>0:
                    ''''
                    choose the sample randomly. without any SDP computation
                    '''
                        
#                        idx=pl_intra.index(min(pl_intra))
                        r=random.choice(pl_intra)
                        
#                        print(sel_intra)
                        idx=pl_intra.index(r)
                        sel_intra=intra[idx]
                        s_path=pw_intra[idx]
                        triplet=triplet_intra[idx]
                        sampled.append(sel_intra)
                        self.samples.append((pmid,sel_intra[0],pr,label,s_path,triplet,'intra'))
                        
                    # iF I CAN't find the path lenght, by default I take first sentence as sample
                    else:
                        sel_intra=intra[0]
                        
#                        if label==0:
                        
                        pl,pw,d,e1,e2,r=sdp(sel_intra[-1],pair)
                        s_path=pw
                        triplet=[e1,r,e2]
                        sampled.append(sel_intra)
                        self.samples.append((pmid,sel_intra[0],pr,label,s_path,triplet,'intra'))
                elif len(intra)==1:
                    sel_intra=intra[0]
                    

                    pl,pw,d,e1,e2,r=sdp(sel_intra[-1],pair)
                    s_path=pw
                    sampled.append(sel_intra)
                    triplet=[e1,r,e2]
                    self.samples.append((pmid,sel_intra[0],pr,label,s_path,triplet,'intra'))
                

                
                if len(inter)>1:
#                    print(inter)
                    pl_inter=[]
                    pw_inter=[]
                    triplet_inter=[]
                    for item in inter:
#                        if label==0:
                        s1=item[0][-1]
                        s2=item[1][-1]
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

                    
                    if len(pl_inter)>0:

                        r1=random.choice(pl_inter)
                        
#                        print(sel_intra)
                        idx=pl_inter.index(r1)
                        
                        sel_inter=inter[idx]
                        s_path=pw_inter[idx]
                        s_triplet=triplet_inter[idx]
                        sampled.append((sel_inter,pair,label))
#                        print('selected',sel_inter,pair,label)   
                        sel_inter=sel_inter[0][0]+' '+sel_inter[1][0]
#                        sel=sel_inter[0]+sel_inter[2]
                        self.samples.append((pmid,sel_inter,pr,label,s_path,s_triplet,'inter'))
                    else:
                        sel=inter[0]
                        sel_inter=sel[0][0]+' '+sel[1][0]
                        
                        s1=sel[0][-1]
                        s2=sel[1][-1]
#                        print(s1,pair[0])
                        pl1,pw1,d1,e1,e11=get_parsed(s1,pair[0]) 
                        pl2,pw2,d2,e2,e22=get_parsed(s2,pair[1]) 
                        pw2.reverse()
                        pw=pw1+pw2
                        s_path=pw
                        sampled.append((sel_inter,pair,label))
                        s_triplet=[e1,e11,e22,e2]
                        self.samples.append((pmid,sel_inter,pr,label,s_path,s_triplet,'inter'))
                elif len(inter)==1:
                    sel=inter[0]
                    sel_inter=sel[0][0]+' '+sel[1][0]
                    s1=sel[0][-1]
                    s2=sel[1][-1]
#                        print(s1,pair[0])
                    pl1,pw1,d1,e1,e11=get_parsed(s1,pair[0]) 
                    pl2,pw2,d2,e2,e22=get_parsed(s2,pair[1]) 
                    pw2.reverse()
                    pw=pw1+pw2
                    s_path=pw
                    sampled.append((sel_inter,pair,label))
                    s_triplet=[e1,e11,e22,e2]
                    self.samples.append((pmid,sel_inter,pr,label,s_path,s_triplet,'inter'))   
                
                self.t+=len(sampled)
                if label==1:
                    self.pp+=len(sampled)
                else:
                    self.nn+=len(sampled)
#                self.samples=list(set(self.samples))
                self.full_sample.append((full_abstract,pair,label))
                        

                    
        print(len(self.full_sample))
        return self.full_sample,self.samples
    '''
    Creating positive and negative samples from the text. The relation will be binary either 0/1
    '''
    def write_path(self):

        full1=os.path.join(self.opath,'dev_nosdp.tsv')

        with open(full1, 'w',encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t',lineterminator='\n')
            for _,r1 in enumerate(self.samples): 
#                print(r1)
                if len(r1)>0:
#                    pl,pw,d=sdp(r1[0],r1[1])
                    
                    
                    writer.writerow([r1[0],r1[1],r1[2],r1[3],r1[4],r1[5],r1[6]])
                
                    
           

        
def parse_args():
        parser = argparse.ArgumentParser(description="Parse the data")

        
        parser.add_argument('--biocpath', type=str, default='train/',
                            help='Folder name training.')
        
        parser.add_argument('--pubtator', type=str, default='CDR_Data/',
                            help='Folder name pubtator data.')

        

        parser.add_argument('--opath', type=str, default='DATA_EVAL/dev',
                            help='Name of the desired output folder.')

        

        args =  parser.parse_args()
        return args
    
args = parse_args()


m = preprocess_model(args)

data=m.extract_data('CDR_DevelopmentSet.PubTator.txt')
#CDR_DevelopmentSet.PubTator
#CDR_TestSet.PubTator
#CDR_TrainingSet.PubTator
#CDR_sample.PubTator.txt
infos,info_list=m.extract_rel_data()
fullsample,sample=m.formulate_samples()
m.write_path()
