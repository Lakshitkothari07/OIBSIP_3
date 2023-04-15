import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

#Read Data 
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv")
data.head(5)
#So this data doesn't have any null value, now let's look at some of the other important insights to get an idea of what kind of data we're dealing with:


data.isnull().sum()

data.info()

print(data.describe())

data.CarName.unique()
#The price in the dataset is supposed to be the column whose values we need to predict. So let's see the distribution of the values of the price column:

sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sns.distplot(data.price)
plt.show()
#Now let's have a look at the correlation among all the features of this dataset:

print(data.corr())

