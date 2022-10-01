import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)

#st.title("Are you going to Default...!!!")
st.markdown("<h1 style='text-align: center; color: green;'> Are you going to Default...!!! </h1>", unsafe_allow_html=True)
st.sidebar.success("Select a page above.")
st.markdown("This App predicts chances of credit card defaults (Non-payment of credit card bills) using Machine Learning Techniques. The Model is trained using historical data containing various details of clients. For predicting future deaults only following parameters are considered, which plays important role.")
#st.write("The Model is trained using historical data containing various details of clients.")
#st.write("For predicting future deaults only following parameters are considered, which plays important role.")
st.markdown("- Repayment status in September, 2005 (-2=no consumption, -1=pay duly, 0=the use of revolving credit, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)")
st.markdown("- Age in years")
st.markdown("- Last month Bill Amount")
st.markdown("- Amount of given credit in NT dollars (includes individual and family/supplementary credit)")
st.markdown("- First previous month Bill Amount (NT dollar)")
st.markdown("- Second previous month Bill Amount (NT dollar)")

st.markdown("<h3 style='text-align: left; color: red;'> Data Analysis Results: </h3>", unsafe_allow_html=True)
st.write(" From the Analysis of historical data has been found that 77.88% of clients are Non-defaulters and only 22.12% are only the Defaulters")
image = Image.open('blob.jpg')
st.image(image, caption='Non-defaulter to Defaulter Proportion')
st.markdown("[Sample data input: 2, 23, 28224, 30000, 29276, 28635]")
