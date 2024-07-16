import streamlit as st
import pandas as pd
import seaborn as sns
import io
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.mixture import GaussianMixture
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D




data = pd.read_csv('market.csv')


def show_cluster(): 
    st.title('Кластеризация') 
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
   

    scaler = MinMaxScaler() 
    normalized_data = scaler.fit_transform(data) 
    data_norm = pd.DataFrame(normalized_data, columns=data.columns)



    # Метод логтя
    st.subheader('Метод логтя')
    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(data_norm)
        distortions.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), distortions, marker='o')
    ax.set_xlabel('Количество кластеров')
    ax.set_ylabel('Расстояние')
    st.pyplot(fig)


# Кластеризация
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(data_norm)

# Кластер
    labels = kmeans.labels_
    data['Cluster'] = labels
 
    st.subheader('Clustered Data')
    st.write(data)

# Распределения кластеров
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Cluster', data=data, ax=ax2)
    st.pyplot(fig2)

# Распределения кластеров относительно Трат и Доходов
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=data, x=data["Spent"], y=data["Income"], hue=data["Cluster"], ax=ax3)
    ax3.set_title("Распределения кластеров относительно Трат и Доходов")
    st.pyplot(fig3)

# Boxenplot
    fig4, ax4 = plt.subplots()
    sns.boxenplot(x=data["Cluster"], y=data["Spent"], ax=ax4)
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots()
    sns.boxenplot(x=data["Cluster"], y=data["Income"], ax=ax5)
    st.pyplot(fig5)


    fig6, ax6 = plt.subplots()
    sns.boxenplot(x=data["Cluster"], y=data["Family_Size"], ax=ax6)
    st.pyplot(fig6)


    fig7, ax7 = plt.subplots()
    sns.boxenplot(x=data["Cluster"], y=data["Age"], ax=ax7)
    st.pyplot(fig7)


    fig8, ax8 = plt.subplots()
    sns.boxenplot(x=data["Cluster"], y=data["Living_With"], ax=ax8)
    st.pyplot(fig8)


    st.write("""
Кластер 0:
• Клиент являетесь родителем. 
• В семье не более 4 членов и не менее 2 
• Родители-одиночки входят в эту группу 
• Возраст клиентов относительно старше
""")
 
    st.write("""
Кластер 1 : 
• Клиент не являетесь родителем 
• В семье максимум 2 члена 
• Небольшое большинство пар по сравнению с одинокими людьми 
• Охватывают все возрасты 
• Группа с высоким доходом
""")

    st.write("""
Кластер 2 : 
• Большинство этих людей — родители 
• Максимум 3 человека в семье 
• Относительно моложе
""")

    st.write("""
Кластер 3 : 
• Они определенно являются родителями 
• В семье максимум 5 членов и минимум 2 
• Относительно старше 
• Группа с низкими доходами
""")

