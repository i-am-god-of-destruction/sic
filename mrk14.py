import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,SimpleRNN,Dropout,Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

data = pd.read_csv("mrk0.csv")
x = data.drop("hospital_death", axis=1)
y = data["hospital_death"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = load_model("d2.h5")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
y_pred=model.predict(x_test)
y_pred[y_pred>0.5]=1
y_pred[y_pred<0.5]=0
print(accuracy_score(y_pred,y_test))
print(f1_score(y_test,y_pred))