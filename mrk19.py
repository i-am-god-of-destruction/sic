from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score,f1_score
from tensorflow import keras
# import warnings
# import matplotlib.pyplot as plt
# warnings.filterwarnings("ignore")

data = pd.read_csv("mrk1.csv")
x = data.drop('hospital_death', axis=1)
y = data['hospital_death']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = load_model("hyper.h5")
model.compile(loss='binary_crossentropy',optimizer="adam",metrics=["accuracy"])
model.summary()
fine_tune_epochs = 10
total_epochs =  20 + fine_tune_epochs
#history_fine = model.fit(x_train,y_train,
                         # epochs=total_epochs,
                         # #initial_epoch=history.epoch[-1],
                         # validation_data=(x_test,y_test))
y_pred=model.predict(x_test)
y_pred[y_pred>0.5]=1
y_pred[y_pred<0.5]=0
print(accuracy_score(y_pred,y_test))
print(f1_score(y_test,y_pred))