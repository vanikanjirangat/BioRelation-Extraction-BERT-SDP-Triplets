# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:15:21 2020

@author: Install
"""

import scispacy
import spacy
nlp = spacy.load("en_core_sci_sm")
import networkx as nx
import re
'''
The problem is with multiword entity
'''
def sdp(text,entity):

    doc = nlp(text)
#    print(entity)
#    print(text)
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(doc[ent.start:ent.end])
            
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),'{0}'.format(child.lower_)))
            
    graph = nx.Graph(edges)# Get the length and path
    
   
    entity1 = str(entity[0])
    entity2 = str(entity[1])
    
    e1=entity1.split()[0]
    e2=entity2.split()[0]
    
    entity1=e1
    entity2=e2
    
    token_list=[]
    dep_list=[]
    for token in doc:
        token_list.append(token)
        dep_list.append(token.dep_)
    for i,j in enumerate(dep_list):
        if j=='ROOT':
            predicate=str(token_list[i])
        
        
    for token in token_list:
        
        if entity1 in str(token):
            entity1=str(token).lower()
            
        if entity2 in str(token):
            entity2=str(token).lower()
            
    triple=[entity1,predicate,entity2]      
 
    try:
        n=nx.shortest_path_length(graph, source=entity1, target=entity2)
#        print(n)
        m=nx.shortest_path(graph, source=entity1, target=entity2)
        m=[re.sub(r'[\.]','',x) for x in m]
#        print(m)
#        print('success:',entity)
#    except:
#        pass
#        try:
#            n=nx.shortest_path_length(graph, source=entity2, target=entity1)
#            m=nx.shortest_path(graph, source=entity2, target=entity1)
#            m=[re.sub(r'[\.]','',x) for x in m]
##            print('success:',entity)

    except Exception:
#        print('Caught this error: ', (entity1,entity2))
#        print(text)
        #print(label,text)
        return -1,['nil'],'nil',entity1,entity2,predicate
    
#    for token in doc:
#        dep_rel=(token.head.text.lower(), token.text.lower(), token.dep_.lower())
#        print((re.sub(r'[\.]','',dep_rel[0]),re.sub(r'[\.]','',dep_rel[1]),dep_rel[2]))
    
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

        s=''
        
        for k,item in enumerate(m):
            if k!=len(m)-1:
                s+=item+'--'+deps1[k]+'-->'
            else:
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
#    print('triple',triple)
    return n,m,s1,entity1,entity2,predicate
#
#text='Famotidine is a histamine H2-receptor antagonist used in inpatient settings for prevention of stress ulcers and is showing increasing popularity because of its low cost.'
#entity=('Famotidine','ulcers')
#n,m,s1,t=sdp(text,entity)
##print(t1,s1)
#
#text="The aim of this work is to call attention to the risk of tacrolimus use in patients with SSc."
#entity=('tacrolimus', 'SSc')
#n,m,s1,t=sdp(text,entity)
#
#text="Risk factors and predictors of levodopa induced dyskinesia among multiethnic Malaysians with Parkinson's disease."
#entity=('levodopa', 'dyskinesia')	
#n,m,s1,t=sdp(text,entity)