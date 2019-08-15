#import libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#import the data set
dataset = pd.read_csv('Data.csv')
##variables independientes
X = dataset.iloc[:,:-1].values
##variables dependientes
Y = dataset.iloc[:,3].values



from sklearn.impute import SimpleImputer
imputer =  SimpleImputer(missing_values = np.NaN, strategy='mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])