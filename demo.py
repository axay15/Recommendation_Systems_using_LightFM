# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 22:18:44 2018

@author: AXAY
"""

import numpy as np
import lightfm.datasets as fetch_movielens
from lightfm import LightFM


# fetch data and format it
data = fetch_movielens(min_rating=4.0) # We're only collecting movie with rating of 4.0 or higher
# our data variable is a dictionary

# movielens is a csv file that contains 100k movie ratings from 1k users on 1700 movies
# each user has rated atleast 20 movies on a scale of 1 to 5  


#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# create model 
model = LightFM(loss='warp') 
#warp - Weighted Approximate-Rank Pairwise; uses gradient descent to iteratively find the weights that improve our prediction over time. 
#this is a Content + Collaborative = Hybrid system

#train model
model.fit(data['train'], epochs=30, num_threads=2)
#num_threads = no. of parallel computations

def sample_recommendation(model, data, user_ids):
    # number of users and movies in the training data
    n_users, n_items = data['train'].shape
    # generate recommendations for each user we input
    for user_id in user_ids:
        # movies they already like 
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        #movies our model predicts they'll like
        scores = model.predict(user_id, np.arange(n_items))
        # we are predicting scores for every movie and saving the result in scores variable
        
        # rank them in order of most to least liked
        top_items = data['item_labels'][np.argsort(-scores)]
        
        # print out the results
        print("User %s" % user_id)
        print("      Known Positives:")
        
        for x in known_positives[:3]:
            print("         %s" % x)
            
        print("      Recommended:   ")
        
        for x in top_items[:3]:
            print("         %s" % x)
            
# predict recommendations for 3 sample user_ids
sample_recommendation(model, data, [3, 25, 450])

        
                