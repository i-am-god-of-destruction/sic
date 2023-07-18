import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
# import matplotlib.pyplot as plt
import pickle

data = pd.read_csv("mrk1.csv")
x = data.drop('hospital_death', axis=1)
y = data['hospital_death']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
models = {
    # "Logistic Regression": LogisticRegression(max_iter=1000),
    # "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    # "SVC": SVC(),
    # "KNC": KNeighborsClassifier(),
    # "MLP": MLPClassifier(max_iter=1000)
}

results = {}

for name, model in models.items():
    history = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    results[name] = r2
    print(f"{name}: {r2}")
    print(f"{name}: {f1}")
    filename='trained.sav'
    with open(filename,"wb") as f:
        pickle.dump(model,f)
    #plt.plot(history.history['accuracy'])
   # plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(['train'], loc='upper left')
    #plt.show()