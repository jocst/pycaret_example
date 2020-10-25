from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

# load trained model for predictions
model = load_model('deployment_19102020')

# function for online prediction, returns only 1 value
def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('logo.png')

    st.image(image, use_column_width=False)

    # add sidebar
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch")
    )

    st.sidebar.info('This app is created to predict insurance premium')
    st.sidebar.success('http://www.pycaret.org')

    # web app main page
    st.title("Insurance Premium Prediction App")

    # for online prediction - capture all inputs required for prediction using streamlit widgets
    if add_selectbox == 'Online':
        age = st.number_input('Age', min_value=18, max_value=100, value=33)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('Children', [0,1,2,3,4,5,6])
        if st.checkbox('Smoker'):
            smoker = 'yes'
        else:
            smoker = 'no'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output=""

        # transform input into dataframe
        input_dict = {'age':age, 'sex':sex, 'bmi':bmi, 'children':children, 'smoker':smoker, 'region':region}
        input_df = pd.DataFrame([input_dict])

        # invoke predict function when button is hit.
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output) # predicted value is a financial value

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for prediction", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
