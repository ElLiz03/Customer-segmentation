import pandas as pd
import numpy as np
import plotly.express as px
from plotly.offline import iplot
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
import io
import sys
import joblib
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('market.csv')

# Explore Page 
def show_analys():    
    st.title('Анализ датасета') 
    global data
    buffer = io.StringIO()
    # Сохраняем текущий поток stdout
    old_stdout = sys.stdout
# Перенаправляем stdout в наш буфер
    sys.stdout = buffer
# Возвращаем stdout в его обычное состояние
    sys.stdout = old_stdout
# Получаем вывод из буфера
    info_text = buffer.getvalue()

    data = data[['Income', 'Age', 'Spent', 'Living_With', 'Children', 'Family_Size', 'Is_Parent', 'Education']]

    # Создание тепловой карты (heatmap) для матрицы корреляций 
    st.subheader('Матрица корреляций') 
    correlation_matrix = data.corr() 
    fig, ax = plt.subplots() 
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax) 
    st.pyplot(fig) 



    st.subheader('Модель множественной регрессии') 
    X = data[['Income', 'Family_Size', 'Is_Parent']]
    y = data['Spent']
    X = sm.add_constant(X) 
    # Добавление столбца констант к предикторам
# Построение модели OLS
    model = sm.OLS(y, X).fit()
# Отображение результатов
    st.write(model.summary())




    