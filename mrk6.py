from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow import keras
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("mrk0.csv")
x = data.drop('hospital_death', axis=1)
y = data['hospital_death']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

optimizer='adam'
init='glorot_uniform'
reg = l2(0.001)
model = Sequential()   
model.add(Dense(512,input_shape=(106,),activation='relu',kernel_initializer=init))
model.add(Dense(256,activation='relu',kernel_regularizer=reg,kernel_initializer=init))
model.add(Dropout(.25))
model.add(Dense(128,activation='relu',kernel_regularizer=reg,kernel_initializer=init))
model.add(Dropout(.25))
model.add(Dense(64,activation='relu',kernel_regularizer=reg,kernel_initializer=init))
model.add(Dropout(.25))
model.add(Dense(1,activation='sigmoid',kernel_regularizer=reg,kernel_initializer=init))
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=["accuracy"])
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# models = [create_model(512,256,128,64,dropout=0.25,regu=0.001),
#           create_model(256,128,64,32,dropout=0.25,regu=0.001),
#           create_model(128,64,32,24,dropout=0.25,regu=0.001),
#           create_model(64,32,16,8,dropout=0.25,regu=0.001),
#          ] 
history = model.fit(x_train,y_train, validation_data = (x_test, y_test),epochs=20,callbacks=[early_stopping],batch_size=32)
model.save('m3.h5')
model.save_weights('m3_w.h5')
y_pred=model.predict(x_test)
y_pred[y_pred>0.5]=1
y_pred[y_pred<0.5]=0
print(accuracy_score(y_pred,y_test))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()