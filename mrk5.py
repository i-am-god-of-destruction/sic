from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow import keras
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

data = pd.read_csv("mrk0.csv")
x = data.drop('hospital_death', axis=1)
y = data['hospital_death']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Flatten(input_shape=(106,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(0.01),loss='binary_crossentropy',metrics=["accuracy"])
epochs = 20
batch_size = 32
history = model.fit(x_train,y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)
model.save('m2.h5')
model.save_weights('m2_w.h5')
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