import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('credit card clients.csv')
# separate the data into features and target
#features = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
#features = data.columns[0]
#st.write(features)
#target = pd.Series(iris_data.target)

# split the data into train and test
#x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target)

#pickle_in = open("classifier.pkl", "rb")
#classifier = pickle.load(pickle_in)
target_names = ["Will not Default"," May Default"]
with open('classifier.pkl','rb') as modelFile:
     model = pickle.load(modelFile)

#Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
#repay_stat= 
class StreamlitApp:

    def __init__(self):
        self.model = model

    #def train_data(self):
       # self.model.fit(x_train, y_train)
        #return self.model

    def construct_sidebar(self):

        cols = [col for col in data.columns]

        '''st.sidebar.markdown('<p class="header-style"> Data for Classification</p>',unsafe_allow_html=True)'''
        st.sidebar.markdown("<h1 style='text-align: center; color: red;'> Client Details </h1>", unsafe_allow_html=True)
        
        PAY_0 = st.sidebar.number_input("Last month Repayment status")
        #PAY_0 = st.sidebar.selectbox(f"Select {cols[0]}",sorted(features[cols[0]].unique()))
        #option = st.selectbox('Repayment status in September-2005', ('Email', 'Home phone', 'Mobile phone'))
        AGE = st.sidebar.number_input("Age")
        LIMIT_BAL = st.sidebar.number_input("Amount of given credit ")
        BIL_AMT1 = st.sidebar.number_input("Last month Bill Amount ")
        BIL_AMT2= st.sidebar.number_input("First previous month Bill Amount")
        BIL_AMT3 = st.sidebar.number_input("Second previous month Bill Amount")
        submit = st.sidebar.button("Submit")
        values = [PAY_0, AGE, BIL_AMT1, LIMIT_BAL, BIL_AMT2, BIL_AMT3]
        #st.form_submit_button(label="Submit")
        #st.write(values)
        #number = st.sidebar.number_input('Insert a number')
        return values

    def plot_pie_chart(self, probabilities):
        fig = go.Figure(data=[go.Pie(labels=list(target_names),values=probabilities[0])])
        fig = fig.update_traces(hoverinfo='label+percent',textinfo='value',textfont_size=15)
        return fig

    def construct_app(self):

        self.model
        values = self.construct_sidebar()

        values_to_predict = np.array(values).reshape(1, -1)

        prediction = self.model.predict(values_to_predict)
        #st.write(prediction[0])
        prediction_str = target_names[prediction[0]]
        probabilities = (self.model.predict_proba(values_to_predict))*100

        st.markdown("""
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
            .font-style {
                font-size:25px;
                Color: Green;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        '''st.markdown('<p class="header-style"> Predictions for the Data </p>', unsafe_allow_html=True)'''
        st.markdown("<h1 style='text-align: center; color: red;'> Credit Card Default Prediction </h1>", unsafe_allow_html=True)
        st.markdown("***")
        '''st.markdown('<style>body{background-color: Blue;}Predictions for the Data</style>',unsafe_allow_html=True)'''
        column_1, column_2 = st.columns(2)
        column_1.markdown(f'<p class="font-style" >Prediction </p>', unsafe_allow_html=True)
        column_1.write(f"{prediction_str}")
        column_2.markdown('<p class="font-style" >Probability </p>', unsafe_allow_html=True)
        column_2.write(f"{probabilities[0][prediction[0]]} %")

        fig = self.plot_pie_chart(probabilities)
        st.markdown('<p class="font-style" >Probability Distribution</p>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

        return self


sa = StreamlitApp()
sa.construct_app()