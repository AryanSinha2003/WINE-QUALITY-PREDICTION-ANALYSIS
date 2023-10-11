import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open("C:/Users/Aryan Sinha/Desktop/Wine_pred/trained_model.sav","rb"))


def wine_prediction(input_data):
    input_data=[[7,0.32,0.34,1.3,0.042,20,69,0.9912,3.31,0.65,12]]

    np_array=np.asarray(input_data)

    np_array_reshape=np_array.reshape(-1,1)
    prediction=loaded_model.predict(np_array)
    return(prediction)


def main():

    #giving a title
    st.title("Wine Prediction Web App")

    #getting input data from user
    fixed_acidity=st.text_input("Enter Fixed acidty")
    volatile_acidity=st.text_input("Enter Volatile acidity")
    citric_acid=st.text_input("Enter Ctric acid Content")
    residual_sugar=st.text_input("Enter Residual sugar content")
    chlorides=st.text_input("Enter Chlorides Amount")
    free_sulfur_dioxide=st.text_input("Enter Free sulfur dioxide")
    total_sulfur_dioxide=st.text_input("Enter total sulfur dioxide")
    density=st.text_input("Enter density ")
    pH=st.text_input("Enter pH value")
    sulphates=st.text_input("Enter sulphates")
    alcohol=st.text_input("Enter alcohol")

    #prediction
    prediction= ''

    #creating a button
    if st.button('Wine Quality Prediction Result'):
        prediction=wine_prediction([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol])


    st.success(prediction)

if __name__=="__main__":
    main()

