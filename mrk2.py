import numpy as np 
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score,
							 classification_report,
							 roc_auc_score, roc_curve, auc, precision_recall_curve,
							 confusion_matrix)
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",250)

data = pd.read_csv("/home/akugyo/Downloads/archive/Dataset.csv/Dataset.csv")
data.drop("elective_surgery",axis=1,inplace=True)
details = pd.read_csv("/home/akugyo/Downloads/archive/Data Dictionary.csv")
num_colm = len(data.select_dtypes(include=["number"]).columns)
print(f"Patient Survical Data has {num_colm} numerical columns")
cat_colm = len(data.select_dtypes(include=object).columns)
print(f"Patient Survival Data has {cat_colm} categorical columns")

dicts = {}
res = 0
res1 = 0
res2 = 0
res3 = 0
for col in data.columns:
    missingCount = data[col].isnull().sum()
    percent = (100*missingCount)/data.shape[0]
    if percent>90:
        res += 1
    if percent>80:
        res1 += 1
    if percent>70:
        res2 += 1
    if percent>50:
        res3 += 1
dicts['missing values>90%: '] = res 
dicts['missing values>80%: '] = res1
dicts['missing values>70%: '] = res2 
dicts['missing values>50%: '] = res3
for key,val in dicts.items():
    print(key,val)

res = []
for col in data.columns:
    missingCount = data[col].isnull().sum()
    percent = (100*missingCount)/data.shape[0]
    if percent>50:
        res.append(col)

higherMissingValueColumns = res
for col in  higherMissingValueColumns:
    data.drop(col,axis=1,inplace=True)
data = data[data[['bmi', 'weight', 'height']].isna().sum(axis=1) == 0]
data.drop(['encounter_id','patient_id','hospital_id'],axis=1,inplace=True)
data.drop(['icu_id'],axis=1,inplace=True)
data.drop(['readmission_status'],axis=1,inplace=True)
numeric_cols = data.select_dtypes(include=['number']).columns
categoric_cols = data.select_dtypes(include=object).columns
my_man = []
for i in numeric_cols:
	if len(data[i].unique())>2:
		my_man.append(i)
#print(my_man)
for col in categoric_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

for col in data.columns:
    cols = col 
    meanValue = round(data[col].mean(),1)
    data[col].fillna(meanValue,inplace=True)
# print(data.head())
y = data.index.values
for colm in my_man:
	x = data.describe()
	q1 = x[colm].loc["25%"]
	q3 = x[colm].loc["75%"]
	iqr = q3 - q1
	lower_bound = q1 - (1.5 * iqr)
	upper_bound = q3 + (1.5 * iqr)
	for i in y:
		if data.loc[i,colm] > upper_bound:
		    data.loc[i,colm]= upper_bound
		if data.loc[i,colm] < lower_bound:
		    data.loc[i,colm]= lower_bound

data.to_csv('mrk0.csv', index = True)