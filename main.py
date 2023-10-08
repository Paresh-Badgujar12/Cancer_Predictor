import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

file = pickle.load(open("model.pkl", 'rb'))

Breast_cancer_dataset = pd.read_csv("breast-cancer.csv")
x = Breast_cancer_dataset.iloc[:, 2:33].values
y = Breast_cancer_dataset.iloc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=45)
sd = StandardScaler()
x_train = sd.fit_transform(x_train)
x_test = sd.fit_transform(x_test)
model = RandomForestClassifier()
model = model.fit(x_train, y_train)

sb = st.sidebar.radio(options = ['Home','Cancer Predictor','About me'], label = ":red[Navigation Bar]")

if sb == 'Home':
    st.title(":blue[What is Breast Cancer?]")

    im, content = st.columns([1,2])

    im.image("breast-cancer.jpg", "Early diagnosis")
    im.image("Shutterstock_723180226-1200x1200.jpg",'Saves the life')

    content.write("""
Breast cancer is the most common cancer amongst women in India.


Breast cancer occurs when some breast cells begin to grow abnormally. These cells divide more rapidly than healthy cells and continue to gather, forming a lump or mass. The cells may spread (metastasize) to the lymph nodes or other parts of the body. Breast cancer mostly begins with cells in the milk-producing ducts (invasive ductal carcinoma) or in the glandular tissue called lobules (invasive lobular carcinoma) or in other cells or tissue within the breast.


Researchers have identified hormonal, lifestyle and environmental factors that may increase the risk of breast cancer. But it is unclear why some women who have no risk factors develop cancer, yet others with risk factors never do. It is likely that breast cancer is caused by a complex interaction of genetic makeup and environment factors.


Studies say that over 1,70,000 new breast cancer cases are likely to develop in India by 2020. According to research, 1 in every 28 women is likely to get affected by the disease. While breast cancer occurs almost entirely in women, around 1-2% men are likely to get affected, too.
    """)

if sb == 'Cancer Predictor':
    st.sidebar.write(":red[Tumor Measurements]")
    st.title(":blue[Breast Cancer Diagnosis]")
    
    l = [("radius_mean","radius_mean1"), ("texture_mean", "texture_mean1"),
         ("perimeter_mean", "perimeter_mean1"), ("area_mean", "area_mean1"), ("smoothness_mean", "smoothness_mean1"),
         ("compactness_mean", "compactness_mean1"), ("concavity_mean", "concavity_mean1"),
         ("concave points_mean", "concave_points_mean1"), ("symmetry_mean", "symmetry_mean1"),
         ("fractal_dimension_mean", "fractal_dimension_mean1"), ("radius_se", "radius_se1"), ("texture_se", "texture_se1"),
         ("perimeter_se", "perimeter_se1"), ("area_se", "area_se1"), ("smoothness_se", "smoothness_se1"),
         ("compactness_se", "compactness_se1"), ("concavity_se", "concavity_se1"), ("concave points_se", "concave_points_se1"),
         ("symmetry_se", "symmetry_se1"), ("fractal_dimension_se", "fractal_dimension_se1"), ("radius_worst", "radius_worst1"),
         ("texture_worst", "texture_worst1"), ("perimeter_worst", "perimeter_worst1"), ("area_worst", "area_worst1"),
         ("smoothness_worst", "smoothness_worst1"), ("compactness_worst", "compactness_worst1"),
         ("concavity_worst", "concavity_worst1"), ("concave points_worst", "concave_points_worst1"),
         ("symmetry_worst", "symmetry_worst1"), ("fractal_dimension_worst", "fractal_dimension_worst1")]

    dic = {}

    for i, j in l:
        dic[j] = st.sidebar.slider(i, min_value= float(0), max_value=float(Breast_cancer_dataset[i].max()))

    ar = np.array(list(dic.values())).reshape(1,-1)
    ar = sd.transform(ar)
    pr = model.predict(ar)



    if st.button(":green[Start Diagnosis]"):
        if pr[0] == 0:
            st.success("The patient has benign cancer")
        else:
            st.error("The patient has malignant cancer")

if sb == 'About me':
    st.header("MY PROFILE", divider=True)

    pic, inf = st.columns(2)
    pic.image("Paresh.jpg", width=200, use_column_width=True)

    inf.markdown("""
    **Name:** Paresh Umesh Badgujar

    **Education:** Master of Pharmacy (Pharmaceutics)

    **Institute:** Indian Institute of Technology (B.H.U.), Varanasi

    **Skills:**  
    ***Languages:*** Python for Data Science, SQL  
    ***Tools and Technologies:*** PowerBI, MS Office (Excel, PowerPoint, Word), Pandas, Numpy, Matplotlib, Seaborn, Sklearn, Streamlit, Applied Mathematics, Statistics, ML Algorithms  
    ***Soft skills:*** Leadership, Tech savvy, Analytical Thinking and Problem Solving ability

    **Area of interest:** Data Analytics, Machine learning, Healthcare Automations

    **M:** 9049126060

    **Email:** pubadgujar2001@gmail.com 
    """)