import streamlit as st
import joblib
import numpy as np
import pandas as pd

# # Load your trained model
model_ridge = joblib.load('Ridge_model_V3.sav')

genre_mean_GSI_value = {
    'Action': 2.777,
    'Comedy': 2.034,
    'Drama': 2.426,
    'Family': 2.731,
    'Horror': 2.118,
    'Suspense/Thriller' : 1.920,
    'Others' : 1.397
}

genre_mapping = {
    'Action': 50,
    'Comedy': 14,
    'Drama': 17,
    'Family': 33,
    'Horror': 28,
    'Suspense/Thriller' : 19,
    'Others' : 3
}


# # Create the Streamlit interface
st.title('Opening Box Office Prediction')
st.write('Please give inputs to various factors.')

Definite_Aware = st.number_input("Enter Definite_aware for week",min_value=0.0, max_value=1.0, value=0.0, step=0.001,format="%.3f")
Unaided_total = st.number_input("Enter Unaided total for week",min_value=0.0, max_value=1.0, value=0.0, step=0.001,format="%.3f")
First_Choice = st.number_input("Enter First Choice for week",min_value=0.0, max_value=1.0, value=0.0, step=0.001,format="%.3f")
Weekly_index_volume = st.number_input("Enter Cumulative Weekly index volume",min_value=0.0, max_value=10000.0, value=0.0, step=0.01)
Total_Awarenes = st.number_input("Enter Total_awareness for week",min_value=0.0, max_value=1.0, value=0.0, step=0.001,format="%.3f")
Unaided_Intent = st.number_input("Enter Unaided Intent for week",min_value=0.0, max_value=1.0, value=0.0, step=0.001,format="%.3f")
Genre = st.selectbox('Select a genre:', list(genre_mapping.keys()))
Genre_value = genre_mapping[Genre]
mean_index = genre_mean_GSI_value[Genre]


Definite_Aware_cbrt = np.cbrt(Definite_Aware)
Unaided_total_sqrt = np.sqrt(Unaided_total)
First_Choice_sqrt = np.sqrt(First_Choice)
Weekly_index_volume_sqrt = np.sqrt(Weekly_index_volume)
Search_Index_Diff = Weekly_index_volume_sqrt - mean_index
Interaction_UI_TA = Total_Awarenes * Unaided_Intent


# # Make predictions
Prediction = model_ridge.predict([[Definite_Aware_cbrt, Unaided_total_sqrt,First_Choice_sqrt,
                                   Weekly_index_volume_sqrt,Total_Awarenes,Genre_value,
                                   Search_Index_Diff,Interaction_UI_TA]])

Prediction_final = np.exp(Prediction)
 # Display predictions
st.write('Predicted OBO', int(Prediction_final))

