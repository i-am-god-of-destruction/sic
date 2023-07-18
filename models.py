from tensorflow.keras.models import Sequential,load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from numpy import array,mean
from DeepEnsemble.DeepEnsemble import DeepEnsembler

models = list()
model1=load_model("m4.h5")
models.append(model1)
model2=load_model("m5.h5")
models.append(model2)

data = pd.read_csv("mrk1.csv")
data=data.iloc[:50000,:]
x = data.drop('hospital_death', axis=1)
y = data['hospital_death']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred = np.array([y_pred1,y_pred2])
Y_test = np.array(y_test)
Ensembler = DeepEnsembler(y_pred, Y_test, type="Weighted", predThreshold=0.6, metrics="accuracy")
score,Y_pred_ensembled = Ensembler.WeightedClassifier()
print(score)

Ensembler = DeepEnsembler(Y_pred, Y_test, type="Voting", predThreshold=0.6, metrics="accuracy")
score,Y_pred_ensembled = Ensembler.VotingClassifier()
print(score)

Ensembler = DeepEnsembler(Y_pred, Y_test, type="Stacking", predThreshold=0.6, metrics="accuracy")
score,Y_pred_ensembled = Ensembler.StackingClassifier()
print(score)