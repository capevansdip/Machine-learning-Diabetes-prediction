import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

#loading the dataset
df = pd.read_csv('datasets_33873_44826_diabetes.csv')

#renaming 'DiabetesPedigreeFunction' to 'DPF'
df=df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

#replacing 0 to NaN 
df_copy = df.copy(deep=True)
df_copy[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI']]=df_copy[['Pregnancies', 'Glucose', 'BloodPressure',
                    'SkinThickness', 'Insulin','BMI']].replace(0,np.NaN)

#fill NaN value with mean, median of the corresponding columns
df_copy['BMI'].fillna(df_copy['BMI'].mean(),inplace=True)
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(),inplace=True)
df_copy['Pregnancies'].fillna(df_copy['Pregnancies'].median(),inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(),inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].mean(),inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(),inplace=True)

# Model Selection

X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#features scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# Model building

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1,random_state=0)
clf.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(clf, open(filename, 'wb'))