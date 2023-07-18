import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,GRU,Dropout,Dense
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

model = Sequential()
model.add(GRU(256,return_sequences=True, input_shape=(106,1)))
model.add(GRU(128,return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(64,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=10)
model.save("d2.h5")
y_pred = model.predict(x_test)
y_pred[y_pred>0.5] = 1
y_pred[y_pred<0.5] = 0
print(accuracy_score(y_pred, y_test))
print(f1_score(y_test, y_pred))
