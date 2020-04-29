# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:09:09 2020

@author: midas
"""

import pandas as pd

def fleiss_kappa(ratings, n):
    '''
    Computes the Fleiss' kappa measure for assessing the reliability of 
    agreement between a fixed number n of raters when assigning categorical
    ratings to a number of items.
    
    Args:
        ratings: a list of (item, category)-ratings
        n: number of raters
    Returns:
        the Fleiss' kappa score
    
    See also:
        http://en.wikipedia.org/wiki/Fleiss'_kappa
    '''
    items = set()
    categories = set()
    n_ij = {}
    
    for i, c in ratings:
        items.add(i)
        categories.add(c)
        n_ij[(i,c)] = n_ij.get((i,c), 0) + 1
    
    N = len(items)
    
    p_j = dict(((c, sum(n_ij.get((i, c), 0) for i in items) / (1.0 * n * N)) for c in categories))
    P_i = dict(((i, (sum(n_ij.get((i, c), 0) ** 2 for c in categories) - n) / (n * (n - 1.0))) for i in items))

    P_bar = sum(P_i.values()) / (1.0 * N)
    P_e_bar = sum(value ** 2 for value in p_j.values())
    
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    
    return kappa


#


data= pd.read_csv('Total_data_annotated.csv')

Mithun=data['Mithun']
Punyajoy=data['Punyajoy']
Soumyadeep=data['Soumyadeep']

ratings=[]

for i in range(0,len(Mithun)):
    ratings.append(((i+1),Mithun[i]))
    ratings.append(((i+1),Soumyadeep[i]))
    ratings.append(((i+1),Punyajoy[i]))

print("Mithun & Soumyadeep & Punyajoy")
print(fleiss_kappa(ratings, 3))


ratings=[]

for i in range(0,len(Mithun)):
    ratings.append(((i+1),Mithun[i]))
    ratings.append(((i+1),Punyajoy[i]))
print("Mithun & Punyajoy")
print(fleiss_kappa(ratings, 2))


ratings=[]

for i in range(0,len(Mithun)):
    ratings.append(((i+1),Mithun[i]))
    ratings.append(((i+1),Soumyadeep[i]))

print("Mithun & Soumyadeep")
print(fleiss_kappa(ratings, 2))


ratings=[]

for i in range(0,len(Mithun)):
    ratings.append(((i+1),Soumyadeep[i]))
    ratings.append(((i+1),Punyajoy[i]))

print("Soumyadeep & Punyajoy")
print(fleiss_kappa(ratings, 2))
