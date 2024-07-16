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



df = pd.read_csv("market.csv")
df2 = pd.read_excel("marketing_campaign.xlsx")

def show_visual():    
    st.title('Визуализация датасета') 
    global df
    global df2
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

     #Гистгорамма доходов
    st.subheader('Гистограмма доходов') 
    fig, ax = plt.subplots()
    ax.hist(df['Income'], bins=30, edgecolor='black')
    ax.set_xlabel('Зарплата')
    ax.set_ylabel('Частота')
    st.pyplot(fig)


     #Гистгорамма трат
    st.subheader('Гистограмма трат') 
    fig, ax = plt.subplots()
    ax.hist(df['Spent'], bins=30, edgecolor='black')
    ax.set_xlabel('Траты')
    ax.set_ylabel('Частота')
    st.pyplot(fig)


    # Процентное соотношение людей по уровню образования
    st.subheader('Процентное соотношение людей по уровню образования') 
    obraz_counts = df2['Education'].value_counts(normalize=True) * 100 
    st.bar_chart(obraz_counts)


        # Процентное соотношение размеров семьи
    st.subheader('Процентное соотношение людей по уровню образования') 
    family_counts = df['Family_Size'].value_counts(normalize=True) * 100 
    st.bar_chart(family_counts)


    df2["Spent"] = df2["MntWines"]+ df2["MntFruits"]+ df2["MntMeatProducts"]+ df2["MntFishProducts"]+ df2["MntSweetProducts"]+ df2["MntGoldProds"]
    
    #График соотношения доходов и уровня образования
    fig, ax = plt.subplots()
    sns.barplot(x=df2['Education'], y=df2['Income'])
    plt.title('Образование vs Доходы')
    plt.xticks(rotation=90)
    st.pyplot(fig)


    #График соотношения расходов и количества членов семьи
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['Family_Size'], y=df['Spent'])
    plt.title('Размер семьи vs Расходы')
    plt.xticks(rotation=90)
    st.pyplot(fig)

