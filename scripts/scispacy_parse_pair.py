# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:15:21 2020

@author: Install
"""

import scispacy
import spacy
from spacy import displacy
from nltk import Tree
from IPython.core.display import display, HTML
import re
nlp = spacy.load("en_core_sci_sm")
#text=u'Convulsions that occur after DTaP are caused by a fever, and fever may cause headache.'
#text='Paul Allen started a company and named Vern Raburn its President.'
#text2='The company to be called Paul Allen Group will be based in Bellevue Washington. '


#text1="The ocular myasthenia associated with combination therapy of IFN and ribavirin for CHC is very rarely reported; therefore, we present this case with a review of the various eye complications of IFN therapy."
#
#text2="Ophthalmologic examinations showed ptosis on the right upper lid and restricted right eye movement without any other neurological signs."
#text='To determine mitochondrial events from HAART in vivo, 8-week-old hemizygous transgenic @Disease@ mice (NL4-3Delta gag/pol; TG) and wild-type FVB/n littermates were treated with the HAART combination of @OTHER_C@, @Chemical@, and @OTHER_C@ or vehicle control for 10 days or 35 days.'
#(4,7,6,8)
#('ribavirin', 'ocular myasthenia')
#def get_tokens(text):
#    doc = nlp(text)
#    token_list=[]
#    dep_list=[]
#    for token in doc:
#        token_list.append(token)
#        dep_list.append(token.dep_)
#    return token_list,dep_list
#
#
#token_list1,dep_list1=get_tokens(text1)
#token_list2,dep_list2=get_tokens(text2)

import networkx as nx


def get_parsed(text,a):
    #MERGING THE MULTI ENTITIES
    doc = nlp(text)
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(doc[ent.start:ent.end])
            
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),'{0}'.format(child.lower_)))
            
    graph = nx.Graph(edges)# Get the length and path

    tok=[]
    dep=[]
    for token in doc:
        tok.append(token)
        dep.append(token.dep_)
    #entity1 = '@Chemical@'
    #entity2 = '@Disease@'
    
    for i,t in enumerate(dep):
        if t=='ROOT':
            term=tok[i]
    print(term)
    entity1=a
    entity2 = str(term)
    e1=entity1.split()[0]
    e2=entity2.split()[0]
    #
    #print('???',e1,e2)
    entity1=e1
    entity2=e2
    
    
#    print(entity1,entity2)
        #e1=entity1.split()
        #e2=entity2.split()
    
    
            
    for token in tok:
        
        if entity1 in str(token):
            entity1=str(token).lower()
            
        if entity2 in str(token):
            entity2=str(token).lower()   
            
#    print('##',entity1,entity2)
    try:
        n=nx.shortest_path_length(graph, source=entity1, target=entity2)
        
        m=nx.shortest_path(graph, source=entity1, target=entity2)
        m=[re.sub(r'[\.]','',x) for x in m]
#        print(m)
    except Exception:
#        print('Caught this error: ' + repr(error))
        #print(label,text)
        return 100000,['nil'],'nil',entity1,entity2
    deps1=[]
    
    r=[]
    
    
    for token in doc:
        #print((token.head.text, token.text, token.dep_))
        dep_rel=(token.head.text.lower(), token.text.lower(), token.dep_.lower())
        dep_rel1=(re.sub(r'[\.]','',dep_rel[0]),re.sub(r'[\.]','',dep_rel[1]),dep_rel[2])
        
        
        for i,item in enumerate(m):
            if i<len(m)-1:
                if m[i] in dep_rel1:
                    if m[i+1] in dep_rel1:
#                        print(dep_rel1)
                        j=(m[i],m[i+1])
                        
                        #print(m[i],m[i+1],dep_rel1[-1])
                        if j not in r:
                            deps1.append(dep_rel1[-1])
                            r.append(j)
                
                    
#    print(deps1)
    if len(deps1)>0:                
        g=[]
        s=''
        
        for k,item in enumerate(m):
            if k!=len(m)-1:
                g.append(item)
                s+=item+'--'+deps1[k]+'-->'
            else:
                g.append(item)
                s+=item
       
        s1=[]
        for k,item in enumerate(m):
            if k==0:
                s1.append(['start'+' '+item+' '+deps1[k]])
            elif k!=len(m)-1:
                s1.append([deps1[k-1]+' '+item+' '+deps1[k]])
            else:
                s1.append([deps1[k-1]+' '+item+' '+'end'])
        
    else:
        s=m[0]
        s1=[m]
    
    return n,m,s1,entity1,entity2


#
s1='The liver biochemistries normalized after stopping the Fluvastatin.'
n,m,s1,entity1,entity2=get_parsed(s1,'Fluvastatin')
#a1='ribavirin'
#a2='restricted right eye movement'
#n1,m1,s1,e1,p=get_parsed(text1,a1)
##print(n1,m1,s1,e1)
#n2,m2,s2,e2,pp=get_parsed(text2,a2)
#m2.reverse()
##print(n2,m2,s2,e2)
#
#ent1=e1
#ent2=e2
#m=m1+m2
#path=m[1:len(m)-1]
#pl=len(path)
#print(ent1,m,ent2)
#print('sdp_lenght',pl)

#doc = nlp(u'Convulsions that occur after DTaP are caused by a fever.')
#print('sentence:'.format(doc))# Load spacy's dependency tree into a networkx graph
#edges = []
#for token in doc:
#    for child in token.children:
#        edges.append(('{0}'.format(token.lower_),
#                      '{0}'.format(child.lower_)))
#graph = nx.Graph(edges)# Get the length and path
#entity1 = 'Convulsions'.lower()
#entity2 = 'fever'
#print(nx.shortest_path_length(graph, source=entity1, target=entity2))
#print(nx.shortest_path(graph, source=entity1, target=entity2))
