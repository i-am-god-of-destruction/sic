import numpy as np 
import pandas as pd
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score,
							 classification_report,
							 roc_auc_score, roc_curve, auc, precision_recall_curve,
							 confusion_matrix)
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
#import shap

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",250)

data = pd.read_csv("/home/akugyo/Downloads/archive/Dataset.csv/Dataset.csv")
#print(data)
data.drop("elective_surgery",axis=1,inplace=True)
#print(data)
details = pd.read_csv("/home/akugyo/Downloads/archive/Data Dictionary.csv")
#print(details)
#print(data.describe())
num_colm = len(data.select_dtypes(include=["number"]).columns)
print(f"Patient Survical Data has {num_colm} numerical columns")
cat_colm = len(data.select_dtypes(include=object).columns)
print(f"Patient Survival Data has {cat_colm} categorical columns")
#print(data.isnull().sum())

# def nullValued(df):
#     data1 = pd.DataFrame(columns=['Col','Count','Percent'])
#     for col in df.columns:
#         missingCount = df[col].isnull().sum()
#         if missingCount>0:
#             #data1=pd.concat(data1,{'Col':col,'Count':missingCount,'Percent':(100*missingCount)/data1.shape[0]},ignore_index=True)
#             data1.loc[len(data1)] = pd.Series({'Col':col,'Count':missingCount,'Percent':(100*missingCount)/data1.shape[0]})

#     return data1.sort_values(by=['Count'],ascending=False)
    
# #print(nullValued(data))

def nullValued(df):
    dicts = {}
    res = 0
    res1 = 0
    res2 = 0
    res3 = 0
    for col in df.columns:
        missingCount = df[col].isnull().sum()
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
    
nullValued(data)

def nullValuedColumn(df):
    res = []
    data1 = pd.DataFrame(columns=['Col','Count','Percent'])
    for col in df.columns:
        missingCount = df[col].isnull().sum()
        percent = (100*missingCount)/data.shape[0]
        if percent>50:
            res.append(col)
    
    return res
        
higherMissingValueColumns = nullValuedColumn(data)
for col in  higherMissingValueColumns:
    data.drop(col,axis=1,inplace=True)
print("Remaining columns are: ",data.shape[1])
data = data[data[['bmi', 'weight', 'height']].isna().sum(axis=1) == 0]
print("Before removal: ",data.shape)
data.drop(['encounter_id','patient_id','hospital_id'],axis=1,inplace=True)
print("After removal: ",data.shape)
fig = px.histogram(data[['age','gender','hospital_death','bmi']].dropna(), x="age", y="hospital_death", color="gender",
                   marginal="box", 
                   hover_data=data[['age','gender','hospital_death','bmi']].columns)
#fig.show()
sns.countplot(data=data,y='hospital_death',palette='Dark2')  
#plt.show()
age_of_female_deaths = data[data["gender"]=="F"][["age", "hospital_death"]].groupby("age").mean().reset_index()
age_of_male_deaths = data[data["gender"]=="M"][["age", "hospital_death"]].groupby("age").mean().reset_index()

fig = make_subplots()
fig.add_trace(go.Scatter(x=age_of_male_deaths['age'],y=age_of_male_deaths['hospital_death'],name='Male Patients'))
fig.add_trace(go.Scatter(x=age_of_female_deaths['age'],y=age_of_female_deaths['hospital_death'],name='Female Patients'))
fig.update_layout(title_text="<b>Average hospital death probability<b>")
fig.update_xaxes(title_text="<b>Patient Age<b>")
fig.update_yaxes(title_text="<b>Avg Hospital death<b>")
#fig.show()
data.drop(['icu_id'],axis=1,inplace=True)
data.drop(['readmission_status'],axis=1,inplace=True)
columns = ["hospital_admit_source", "icu_admit_source", "icu_stay_type", "aids", "leukemia", "immunosuppression"]
plt.figure()
number = 1 
for col in columns:
    if number<=len(columns):
        ax1 = plt.subplot(3,3,number)
        sns.countplot(data=data,y=col,palette='Dark2')
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.title(col.title(), fontsize=14)
        plt.xlabel('')
        plt.ylabel('')
    number += 1 
plt.tight_layout()
plt.show()

print("After a bit of cleaning: ")
print("------------------------")
numeric_cols = data.select_dtypes(include=['number']).columns
print(f"Patient Survival data has {len(numeric_cols)}  numerical columns")
categoric_cols = data.select_dtypes(include=object).columns
print(f"Patient Survival data has {len(categoric_cols)} categorical columns")
encoded_data = deepcopy(data)
encoded_data1 = deepcopy(data)
for col in categoric_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

weight_data = data[['weight','bmi','hospital_death']]
weight_data['weight'] = weight_data['weight'].round(0)
weight_data['bmi'] = weight_data['bmi'].round(0)
weight_death = weight_data[['weight','hospital_death']].groupby('weight').mean().reset_index()
bmi_death = weight_data[['bmi','hospital_death']].groupby('bmi').mean().reset_index()
fig = make_subplots(rows=1,cols=2,shared_yaxes=True)
fig.add_trace(go.Scatter(x=weight_death['weight'],y=weight_death['hospital_death'],name='Weight'))
fig.add_trace(go.Scatter(x=bmi_death['bmi'],y=bmi_death['hospital_death'],name='BMI'))
fig.update_layout(title_text="<b>Impact of BMI & Weights over patients<b>")
fig.show()

ICU_type = data[['icu_type','age','hospital_death']]
ICU_type['icu'] = ICU_type['icu_type'].replace({'CTICU':'CCU-CTICU','Cardiac ICU':'CCT-CTICU','CTICU':'CCT-CTICU','CSICU':'SICU'})
ICU_data = ICU_type.groupby(['icu_type','age']).mean().reset_index()
ICU_data['count'] = ICU_type.groupby(['icu_type','age']).count().reset_index()['hospital_death']
fig = px.scatter(ICU_data,x='age',y='hospital_death',size='count',color='icu_type',hover_name='icu_type',log_x=False,size_max=60,)
fig.update_layout(title_text="<b>Survival Rate at different types of ICU<b>")
fig.update_yaxes(title_text="<b>Avg hospital_death<b>")
fig.update_xaxes(title_text="<b>Age<b>")
fig.show()

print(encoded_data.isnull().sum().sum())
def medianImpute(data,col,medianValue):
    data[col].fillna(medianValue,inplace=True)
    return data
for col in encoded_data.columns:
    cols = col 
    medianValue = data[col].median()
    medianImpute(data,cols,medianValue)
outlier_colms =data.columns
ibm_df1 = data.copy()

def handle_outliers(df, colm):
    '''Change the values of outlier to upper and lower whisker values '''
    q1 = df.describe()[colm].loc["25%"]
    q3 = df.describe()[colm].loc["75%"]
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for i in range(len(df)):
        if df.loc[i,colm] > upper_bound:
            df.loc[i,colm]= upper_bound
        if df.loc[i,colm] < lower_bound:
            df.loc[i,colm]= lower_bound
    return df
    
for colm in numeric_cols:
    ibm_df1 = handle_outliers(ibm_df1, colm)