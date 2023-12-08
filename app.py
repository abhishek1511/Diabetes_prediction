import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

#creating a function for prediction
def diabetesPrediction(inputdata):
    #convert the input data to numpy array
    inputdata_as_nparray = np.asarray(inputdata)

    #reshaping the array as we are predciting for 1 instance
    input_data_reshape = inputdata_as_nparray.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction[0])

    if (prediction[0] == 0):
        return "The patience is not diabetic"
    else:
        return "The patience is diabetic"
    

def main():

    #Title for the web page
    st.title("Diabetes Prediction")
    
    #Getting the input data from the users

    Pregnancies = st.text_input("Number of pregancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of the person")

    # Prediction
    diagonsis = ''

    # Creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagonsis = diabetesPrediction([Pregnancies, Glucose, BloodPressure, 
                                        SkinThickness, Insulin, BMI, 
                                        DiabetesPedigreeFunction, Age])
    st.success(diagonsis)


if __name__ == '__main__':
    main()