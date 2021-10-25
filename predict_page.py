import pickle
import numpy as np
import streamlit as st

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
data = load_model()
regressor = data['model']
le_country = data['le_country']
le_education = data['le_education']
le_employment = data['le_employment']
le_gender = data['le_gender']
le_opsys = data['le_opsys']
le_orgsize = data['le_orgsize']

def show_predict_page():
    st.title("""Software Developer Salary Prediction from stack overflow dataset 2021""")
    st.write("""#### by Aravindh Muthuswamy""")
    st.write("""#### we need some information to predict the salary""")
    countries = {
        "United States of America",
        "India",
        "Germany",
        "United Kingdom of Great Britain and Northern Ireland",
        "Canada",
        "France",
        "Brazil",
        "Spain",
        "Netherlands",
        "Australia",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
        "Turkey",
        "Switzerland",
        "Israel",
        "Norway"          
    }
    education = {
        "Less than a Bachelors",
        "Bachelor’s degree", 
        "Master’s degree",
        "Post grad"
        
    }
    employment = {
        "Full Time", 
        "Part Time", 
        "Freelancer", 
        "Prefer not to say",
        "Retired"
    }
    gender = {
        "Male", 
        "Female", 
        "Other", 
        "Unspecified"
    }
    opsys = {
        "Linux", 
        "Windows", 
        "MacOS", 
        "Other", 
        "BSD"
    }
    orgsize = {
        "Small",
        "Medium",
        "Other",
        "Huge",
        "Extra Large",
        "Tiny",
        "Large",
        "Freelancer",
    }
    countrylvl = st.selectbox("Country", countries)
    educationlvl = st.selectbox("Education Level", education)
    employmentlvl = st.selectbox('Employment Level', employment)
    genderlvl = st.selectbox('Gender', gender)
    opsyslvl = st.selectbox('Operating System', opsys)
    orgsizelvl = st.selectbox('Organization Size', orgsize)
    experience = st.slider("Years of Experience", 0, 50, 3)
    ok = st.button("Calculate salary")
    # st.write(X[: 2])
    if ok:
        X = np.array([[countrylvl, educationlvl, experience, employmentlvl, genderlvl, orgsizelvl, opsyslvl]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X[:, 3] = le_employment.transform(X[:, 3])
        X[:, 4] = le_gender.transform(X[:, 4])
        X[:, 5] = le_orgsize.transform(X[:, 5])
        X[:, 6] = le_opsys.transform(X[:, 6])
        X = X.astype(float)
        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")