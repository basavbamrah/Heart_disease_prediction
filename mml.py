#import libraries

import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

training_set= pd.read_csv('heart.csv')  # importing the training set

X=training_set.drop(["target"],axis=1)  #  removing the "target" column from the data set
Y=training_set['target']
model = GaussianNB()
#Lets train the model
model.fit(X,Y)
pickle.dump(model, open('model.pkl','wb'))      #  Storing the trained model under name model.pkl