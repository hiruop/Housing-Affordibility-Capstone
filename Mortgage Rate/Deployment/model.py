import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle


data = pd.read_csv('C:/Users/Hiren/Desktop/ST.Clair/sem4/402-Capstone/housing-affordability-in-canada/Mortage and Interest rates csv/outliers.csv', parse_dates=['Date'])

# Independent Variables
X = data[['Interest Rate', 'HPI_change']]
y = data['Mortgage Rate']
X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
# Let us split our dataset into test and train groups with  8:2 ratio.
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state = 42)

rf_model = RandomForestRegressor()

rf_model.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(rf_model, open("rf_model.pkl", "wb"))

