import pandas as pd
import numpy as np
import plotly.express as px
from plotly.offline import iplot
import seaborn as sns
import matplotlib.pyplot as plt
import io
import sys
import streamlit as st
from sklearn.preprocessing import LabelEncoder


df = pd.read_excel("marketing_campaign.xlsx")

def show_dataset():    
    st.title('Предобработка датасета') 
    global df
    buffer = io.StringIO()
    old_stdout = sys.stdout
# Перенаправляем stdout в наш буфер
    sys.stdout = buffer
# Вызываем info(), вывод пойдет в буфер
    df.info()
# Возвращаем stdout в его обычное состояние
    sys.stdout = old_stdout
# Получаем вывод из буфера
    info_text = buffer.getvalue()
    st.subheader('Информация о датасете')
    st.text(info_text)
    st.subheader('Описание датасета') 
    st.write(df.describe())

    st.subheader('Проверка наличия пропущенных значений') 
    st.write(df.isnull().sum())

   #Заменяем все пропущенные значения этим средним значением
    mean_income = df['Income'].mean()
    df['Income'] = df['Income'].fillna(mean_income)

    #Обработка датасета
    df["Age"] = 2023-df["Year_Birth"]
    df["Spent"] = df["MntWines"]+ df["MntFruits"]+ df["MntMeatProducts"]+ df["MntFishProducts"]+ df["MntSweetProducts"]+ df["MntGoldProds"]
    df["Living_With"]=df["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})
    df["Children"]=df["Kidhome"]+df["Teenhome"]
    df["Family_Size"] = df["Living_With"].replace({"Alone": 1, "Partner":2})+ df["Children"]  
    df["Is_Parent"] = np.where(df.Children> 0, 1, 0)
    df["Education"]=df["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})
    df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})
    to_drop = ["Dt_Customer", "Year_Birth", "ID", "Z_CostContact", "Z_Revenue"]
    df = df.drop(to_drop, axis=1)


    data = df[['Income', 'Age', 'Spent', 'Living_With', 'Children', 'Family_Size', 'Is_Parent', 'Education', 'Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold']].copy()
    
    st.subheader('Вывод нового датасета') 
# Вывод первых трех строк датасета
    first_three_rows = data.head(3)
    st.write(first_three_rows)

   # Описательная статистика нового датасета 
    st.subheader('Описательная статистика нового датасета') 
    description = data.describe()
    st.write(description)


    #Работа с выбросами Возраст
    Q1 = data['Age'].quantile(0.25)
    Q3 = data['Age'].quantile(0.75)
    IQR = Q3 - Q1
# Определим границы для выбросов
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
# Заменим выбросы на медианное значение
    data['Age'] = np.where((data['Age'] < lower_bound) | (data['Age'] > upper_bound), data['Age'].median(), data['Age'])

    #Работа с выбросами Доходы
    Q1 = data['Income'].quantile(0.25)
    Q3 = data['Income'].quantile(0.75)
    IQR = Q3 - Q1
# Определим границы для выбросов
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
# Заменим выбросы на медианное значение
    data['Income'] = np.where((data['Income'] < lower_bound) | (data['Income'] > upper_bound), data['Income'].median(), data['Income'])

    #Работа с выбросами Траты
    Q1 = data['Spent'].quantile(0.25)
    Q3 = data['Spent'].quantile(0.75)
    IQR = Q3 - Q1
# Определим границы для выбросов
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
# Заменим выбросы на медианное значение
    data['Spent'] = np.where((data['Spent'] < lower_bound) | (data['Spent'] > upper_bound), data['Spent'].median(), data['Spent'])


    st.subheader('Описательная статистика датасета после работы с выбросами') 
# Описательная статистика нового датасета
    description = data.describe()
    st.write(description)



    # Работа с категориальными данными
    le = LabelEncoder()
    data['Living_With'] = le.fit_transform(data['Living_With'])
    data['Education'] = le.fit_transform(data['Education'])


    st.subheader('Вывод готового датасета') 
# Вывод первых трех строк готового для дальнейшей работы датастеа
    first_three_rows = data.head(3)
    st.write(first_three_rows)



    # сохранить DataFrame в файл CSV
    data.to_csv('market.csv', index=False)
