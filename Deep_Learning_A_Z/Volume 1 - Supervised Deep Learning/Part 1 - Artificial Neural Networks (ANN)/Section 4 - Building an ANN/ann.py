# Part 1 - Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:, 3:13].values
# print(x)
y = dataset.iloc[:, 13].values
# print(y)

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]

# Splitting the data into traing and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Part 2 - Now let's make the ANN!
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classfier = Sequential()

# Adding the input layer and the first hidden layer
classfier.add(Dense(6,activation='relu',kernel_initializer='glorot_uniform',input_dim=11))

# Adding the second hidden layer
classfier.add(Dense(6,activation='relu',kernel_initializer='glorot_uniform'))

# Adding the output layer
classfier.add(Dense(1,activation='sigmoid',kernel_initializer='glorot_uniform'))


# Compiling the ANN
classfier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the ANN to the Training set
classfier.fit(x_train,y_train,batch_size=10,nb_epoch=100)


# Prediction
y_pred = classfier.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# # Part 2 - Now let's make the ANN!
#
# # Importing the Keras libraries and packages
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
#
# # Initialising the ANN
# classifier = Sequential()
#
# # Adding the input layer and the first hidden layer
# classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#
# # Adding the second hidden layer
# classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#
# # Adding the output layer
# classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#
# # Compiling the ANN
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
# # Fitting the ANN to the Training set
# classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
#
# # Part 3 - Making predictions and evaluating the model
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5)
#
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
