import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

st.title("Titanic Survival Model")

@st.cache_data
def train_model():

    df = pd.read_csv("Titanic-Dataset.csv")

    df.drop(columns="Cabin",inplace=True)

    df["Age"].fillna(value=28,inplace=True)

    df["Embarked"].fillna(value="S",inplace=True)

    df.drop(columns = ['PassengerId','Name','Ticket',"Fare"],inplace=True)

    df["Sex"] = np.where(df["Sex"]=="male",1,0)

    le = LabelEncoder()

    df["Embarked"] = le.fit_transform(df["Embarked"])

    df["Age"] = np.where(df["Age"]>54,54,df["Age"])

    df["Age"] = np.where(df["Age"]<2,2,df["Age"])

    X  = df.drop(columns="Survived")

    y = df["Survived"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y , test_size=0.3 ,random_state=88)

    model = LogisticRegression()

    model.fit(X_train,y_train)

    return model
model = train_model()

st.sidebar.header("Enter passenger details")
pclass = st.sidebar.selectbox("Passenger class",[1,2,3])
sex = st.sidebar.selectbox("Sex",["male","Female"])
age = st.sidebar.slider("Age",2,54,28)
sibsp = st.sidebar.number_input("siblings",0,8,0)

parch = st.sidebar.number_input("parents",0,5,0)

Embarked = st.sidebar.select_slider("Embarked",["C","Q","S"])
sex = 1 if sex == "male" else 0

Embarked_map = {"C":0 , "Q" : 1 ,"S":2}

Embarked  = Embarked_map[Embarked]

input_data = np.array([[pclass , sex , age , sibsp , parch , Embarked]])
if st.button("Predict Survival"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:

        st.success("passenger Survived")

    else:

        st.error("passenger Did not survived")