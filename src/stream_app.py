# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 12:45:28 2024

@author: kexib
"""

import streamlit as st
from ml import inference
from collections import namedtuple
from PIL import Image


st.title('Прогнозирование дефолта')
st.write("Введите параметры займа")

def get_prediction():
    params = [[customer_age, customer_income, employment_duration, loan_amnt,
               loan_int_rate, term_years, cred_hist_length, start_cred_history,
               home_ownership, loan_intent, loan_grade, historical_default]]
    y_pr, probabilities = inference(params)

    # Извлекаем вероятность для класса дефолта (класс 1)
    default_probability = probabilities[0][1]  # Вероятность для класса 1

    if y_pr[0] == 0:
        st.write('Дефолт маловероятен')
    else:
        st.write('Вероятно будет дефолт')

    st.write(f'Вероятность дефолта: {default_probability:.2f}')  # Выводим вероятность

with st.sidebar:
    image = Image.open("final_project_default/datadata/img.jpg")
    st.image(image, use_column_width=True)  # Отображаем изображение
    st.title("Пояснения для заполнения выпадающих элементов")
    st.write("")
    st.write("Статус домовладения:<br>RENT - аренда<br>OWN - собственность<br>MORTGAGE - ипотека<br>OTHER - остальное", unsafe_allow_html=True)
    st.markdown("Цель кредита:<br>EDUCATION - образование<br>MEDICAL - медицина<br>VENTURE - венчурный бизнес<br>PERSONAL - личная<br>DEBTCONSOLIDATION - консолидация (рефинансирование)<br>HOMEIMPROVEMENT - улучшение жилищных условий", unsafe_allow_html=True) 
    st.markdown('')
    st.write('Модель обучена на данных заемщиков Великобритании.<br>Метрики модели<br>precision = 98<br>recall = 92<br>f1 = 0.95', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    customer_age = st.slider('возраст клиента', min_value=18, max_value=80, step=1)
    term_years = st.slider('срок кредита в годах', min_value=1, max_value=30, step=1)
    cred_hist_length = st.slider('продолжительность кредитной истории клиента в годах', min_value=0, max_value=80, step=1)
    start_cred_history = st.slider('возраст при оформлении первого кредита', min_value=16, max_value=80, step=1)
    customer_income = st.text_input('годовой доход клиента')
    employment_duration = st.text_input('продолжительность трудоустройства в месяцах')
    loan_amnt = st.text_input('запрашиваемая сумма кредита')
    loan_int_rate = st.text_input('процентная ставка по кредиту')

with col2:
    home_ownership = st.selectbox('статус домовладения', options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    loan_intent = st.selectbox('цель кредита', options=['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
    loan_grade = st.selectbox('оценка, присвоенная кредиту', options=['A', 'B', 'C', 'D', 'E'])
    historical_default = st.selectbox('Укажите допускал ли ранее Клиент дефолт?', options=['Y', 'N', 'no_info'])

get_prediction()
