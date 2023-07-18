

import numpy as np
import pickle
import streamlit as st
import pandas as pd


def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = input_data

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    print("HEH")
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 1):
      return 'The person is not Servived'
    else:
      return 'The person is Servived'
  
  
  
def main():
    
        # giving a title
    st.title('Servival Prediction Web App')
    
    
    # getting the input data from the user
    
  
    age=st.text_input('age',value=data["age"].values[0])
    bmi=st.text_input('bmi',value=data["bmi"].values[0])
    ethnicity=st.text_input('ethnicity',value=data["ethnicity"].values[0])
    gender=st.text_input('gender',value=data["gender"].values[0])
    height=st.text_input('height',value=data["height"].values[0])
    hospital_admit_source=st.text_input('hospital_admit_source',value=data["hospital_admit_source"].values[0])
    icu_admit_source=st.text_input('icu_admit_source',value=data["icu_admit_source"].values[0])
    icu_stay_type=st.text_input('icu_stay_type',value=data["icu_stay_type"].values[0])
    icu_type=st.text_input('icu_type',value=data["icu_type"].values[0])
    pre_icu_los_days=st.text_input('pre_icu_los_days',value=data["pre_icu_los_days"].values[0])
    weight=st.text_input('weight',value=data["weight"].values[0])
    apache_2_diagnosis=st.text_input('apache_2_diagnosis',value=data["apache_2_diagnosis"].values[0])
    apache_3j_diagnosis=st.text_input('apache_3j_diagnosis',value=data["apache_3j_diagnosis"].values[0])
    apache_post_operative=st.text_input('apache_post_operative',value=data["apache_post_operative"].values[0])
    arf_apache=st.text_input('arf_apache',value=data["arf_apache"].values[0])
    bun_apache=st.text_input('bun_apache',value=data["bun_apache"].values[0])
    creatinine_apache=st.text_input('creatinine_apache',value=data["creatinine_apache"].values[0])
    gcs_eyes_apache=st.text_input('gcs_eyes_apache',value=data["gcs_eyes_apache"].values[0])
    gcs_motor_apache=st.text_input('gcs_motor_apache',value=data["gcs_motor_apache"].values[0])
    gcs_unable_apache=st.text_input('gcs_unable_apache',value=data["gcs_unable_apache"].values[0])
    gcs_verbal_apache=st.text_input('gcs_verbal_apache',value=data["gcs_verbal_apache"].values[0])
    glucose_apache=st.text_input('glucose_apache',value=data["glucose_apache"].values[0])
    heart_rate_apache=st.text_input('heart_rate_apache',value=data["heart_rate_apache"].values[0])
    hematocrit_apache=st.text_input('hematocrit_apache',value=data["hematocrit_apache"].values[0])
    intubated_apache=st.text_input('intubated_apache',value=data["intubated_apache"].values[0])
    map_apache=st.text_input('map_apache',value=data["map_apache"].values[0])
    resprate_apache=st.text_input('resprate_apache',value=data["resprate_apache"].values[0])
    sodium_apache=st.text_input('sodium_apache',value=data["sodium_apache"].values[0])
    temp_apache=st.text_input('temp_apache',value=data["temp_apache"].values[0])
    ventilated_apache=st.text_input('ventilated_apache',value=data["ventilated_apache"].values[0])
    wbc_apache=st.text_input('wbc_apache',value=data["wbc_apache"].values[0])
    d1_diasbp_max=st.text_input('d1_diasbp_max',value=data["d1_diasbp_max"].values[0])
    d1_diasbp_min=st.text_input('d1_diasbp_min',value=data["d1_diasbp_min"].values[0])
    d1_diasbp_noninvasive_max=st.text_input('d1_diasbp_noninvasive_max',value=data["d1_diasbp_noninvasive_max"].values[0])
    d1_diasbp_noninvasive_min=st.text_input('d1_diasbp_noninvasive_min',value=data["d1_diasbp_noninvasive_min"].values[0])
    d1_heartrate_max=st.text_input('d1_heartrate_max',value=data["d1_heartrate_max"].values[0])
    d1_heartrate_min=st.text_input('d1_heartrate_min',value=data["d1_heartrate_min"].values[0])
    d1_mbp_max=st.text_input('d1_mbp_max',value=data["d1_mbp_max"].values[0])
    d1_mbp_min=st.text_input('d1_mbp_min',value=data["d1_mbp_min"].values[0])
    d1_mbp_noninvasive_max=st.text_input('d1_mbp_noninvasive_max',value=data["d1_mbp_noninvasive_max"].values[0])
    d1_mbp_noninvasive_min=st.text_input('d1_mbp_noninvasive_min',value=data["d1_mbp_noninvasive_min"].values[0])
    d1_resprate_max=st.text_input('d1_resprate_max',value=data["d1_resprate_max"].values[0])
    d1_resprate_min=st.text_input('d1_resprate_min',value=data["d1_resprate_min"].values[0])
    d1_spo2_max=st.text_input('d1_spo2_max',value=data["d1_spo2_max"].values[0])
    d1_spo2_min=st.text_input('d1_spo2_min',value=data["d1_spo2_min"].values[0])
    d1_sysbp_max=st.text_input('d1_sysbp_max',value=data["d1_sysbp_max"].values[0])
    d1_sysbp_min=st.text_input('d1_sysbp_min',value=data["d1_sysbp_min"].values[0])
    d1_sysbp_noninvasive_max=st.text_input('d1_sysbp_noninvasive_max',value=data["d1_sysbp_noninvasive_max"].values[0])
    d1_sysbp_noninvasive_min=st.text_input('d1_sysbp_noninvasive_min',value=data["d1_sysbp_noninvasive_min"].values[0])
    d1_temp_max=st.text_input('d1_temp_max',value=data["d1_temp_max"].values[0])
    d1_temp_min=st.text_input('d1_temp_min',value=data["d1_temp_min"].values[0])
    h1_diasbp_max=st.text_input('h1_diasbp_max',value=data["h1_diasbp_max"].values[0])
    h1_diasbp_min=st.text_input('h1_diasbp_min',value=data["h1_diasbp_min"].values[0])
    h1_diasbp_noninvasive_max=st.text_input('h1_diasbp_noninvasive_max',value=data["h1_diasbp_noninvasive_max"].values[0])
    h1_diasbp_noninvasive_min=st.text_input('h1_diasbp_noninvasive_min',value=data["age"].values[0])
    h1_heartrate_max=st.text_input('h1_heartrate_max',value=data["h1_heartrate_max"].values[0])
    h1_heartrate_min=st.text_input('h1_heartrate_min',value=data["h1_heartrate_min"].values[0])
    h1_mbp_max=st.text_input('h1_mbp_max',value=data["h1_mbp_max"].values[0])
    h1_mbp_min=st.text_input('h1_mbp_min',value=data["h1_mbp_min"].values[0])
    h1_mbp_noninvasive_max=st.text_input('h1_mbp_noninvasive_max',value=data["h1_mbp_noninvasive_max"].values[0])
    h1_mbp_noninvasive_min=st.text_input('h1_mbp_noninvasive_min',value=data["h1_mbp_noninvasive_min"].values[0])
    h1_resprate_max=st.text_input('h1_resprate_max',value=data["h1_resprate_max"].values[0])
    h1_resprate_min=st.text_input('h1_resprate_min',value=data["h1_resprate_min"].values[0])
    h1_spo2_max=st.text_input('h1_spo2_max',value=data["h1_spo2_max"].values[0])
    h1_spo2_min=st.text_input('h1_spo2_min',value=data["h1_spo2_min"].values[0])
    h1_sysbp_max=st.text_input('h1_sysbp_max',value=data["h1_sysbp_max"].values[0])
    h1_sysbp_min=st.text_input('h1_sysbp_min',value=data["h1_sysbp_min"].values[0])
    h1_sysbp_noninvasive_max=st.text_input('h1_sysbp_noninvasive_max',value=data["h1_sysbp_noninvasive_max"].values[0])
    h1_sysbp_noninvasive_min=st.text_input('h1_sysbp_noninvasive_min',value=data["h1_sysbp_noninvasive_min"].values[0])
    h1_temp_max=st.text_input('h1_temp_max',value=data["h1_temp_max"].values[0])
    h1_temp_min=st.text_input('h1_temp_min',value=data["h1_temp_min"].values[0])
    d1_bun_max=st.text_input('d1_bun_max',value=data["d1_bun_max"].values[0])
    d1_bun_min=st.text_input('d1_bun_min',value=data["d1_bun_min"].values[0])
    d1_calcium_max=st.text_input('d1_calcium_max',value=data["d1_calcium_max"].values[0])
    d1_calcium_min=st.text_input('d1_calcium_min',value=data["d1_calcium_min"].values[0])
    d1_creatinine_max=st.text_input('d1_creatinine_max',value=data["d1_creatinine_max"].values[0])
    d1_creatinine_min=st.text_input('d1_creatinine_min',value=data["d1_creatinine_min"].values[0])
    d1_glucose_max=st.text_input('d1_glucose_max',value=data["d1_glucose_max"].values[0])
    d1_glucose_min=st.text_input('d1_glucose_min',value=data["d1_glucose_min"].values[0])
    d1_hco3_max=st.text_input('d1_hco3_max',value=data["d1_hco3_max"].values[0])
    d1_hco3_min=st.text_input('d1_hco3_min',value=data["d1_hco3_min"].values[0])
    d1_hemaglobin_max=st.text_input('d1_hemaglobin_max',value=data["d1_hemaglobin_max"].values[0])
    d1_hemaglobin_min=st.text_input('d1_hemaglobin_min',value=data["d1_hemaglobin_min"].values[0])
    d1_hematocrit_max=st.text_input('d1_hematocrit_max',value=data["d1_hematocrit_max"].values[0])
    d1_hematocrit_min=st.text_input('d1_hematocrit_min',value=data["d1_hematocrit_min"].values[0])
    d1_platelets_max=st.text_input('d1_platelets_max',value=data["d1_platelets_max"].values[0])
    d1_platelets_min=st.text_input('d1_platelets_min',value=data["d1_platelets_min"].values[0])
    d1_potassium_max=st.text_input('d1_potassium_max',value=data["d1_potassium_max"].values[0])
    d1_potassium_min=st.text_input('d1_potassium_min',value=data["d1_potassium_min"].values[0])
    d1_sodium_max=st.text_input('d1_sodium_max',value=data["d1_sodium_max"].values[0])
    d1_sodium_min=st.text_input('d1_sodium_min',value=data["d1_sodium_min"].values[0])
    d1_wbc_max=st.text_input('d1_wbc_max',value=data["d1_wbc_max"].values[0])
    d1_wbc_min=st.text_input('d1_wbc_min',value=data["d1_wbc_min"].values[0])
    apache_4a_hospital_death_prob=st.text_input('apache_4a_hospital_death_prob',value=data["apache_4a_hospital_death_prob"].values[0])
    apache_4a_icu_death_prob=st.text_input('apache_4a_icu_death_prob',value=data["apache_4a_icu_death_prob"].values[0])
    aids=st.text_input('aids',value=data["aids"].values[0])
    cirrhosis=st.text_input('cirrhosis',value=data["cirrhosis"].values[0])
    diabetes_mellitus=st.text_input('diabetes_mellitus',value=data["diabetes_mellitus"].values[0])
    hepatic_failure=st.text_input('hepatic_failure',value=data["hepatic_failure"].values[0])
    immunosuppression=st.text_input('immunosuppression',value=data["immunosuppression"].values[0])
    leukemia=st.text_input('leukemia',value=data["leukemia"].values[0])
    lymphoma=st.text_input('lymphoma',value=data["lymphoma"].values[0])
    solid_tumor_with_metastasis=st.text_input('solid_tumor_with_metastasis',value=data["solid_tumor_with_metastasis"].values[0])
    apache_3j_bodysystem=st.text_input('apache_3j_bodysystem',value=data["apache_3j_bodysystem"].values[0])
    apache_2_bodysystem=st.text_input('apache_2_bodysystem',value=data["apache_2_bodysystem"].values[0])
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    heh = [age,
            bmi,
            ethnicity,
            gender,
            height,
            hospital_admit_source,
            icu_admit_source,
            icu_stay_type,
            icu_type,
            pre_icu_los_days,
            weight,
            apache_2_diagnosis,
            apache_3j_diagnosis,
            apache_post_operative,
            arf_apache,
            bun_apache,
            creatinine_apache,
            gcs_eyes_apache,
            gcs_motor_apache,
            gcs_unable_apache,
            gcs_verbal_apache,
            glucose_apache,
            heart_rate_apache,
            hematocrit_apache,
            intubated_apache,
            map_apache,
            resprate_apache,
            sodium_apache,
            temp_apache,
            ventilated_apache,
            wbc_apache,
            d1_diasbp_max,
            d1_diasbp_min,
            d1_diasbp_noninvasive_max,
            d1_diasbp_noninvasive_min,
            d1_heartrate_max,
            d1_heartrate_min,
            d1_mbp_max,
            d1_mbp_min,
            d1_mbp_noninvasive_max,
            d1_mbp_noninvasive_min,
            d1_resprate_max,
            d1_resprate_min,
            d1_spo2_max,
            d1_spo2_min,
            d1_sysbp_max,
            d1_sysbp_min,
            d1_sysbp_noninvasive_max,
            d1_sysbp_noninvasive_min,
            d1_temp_max,
            d1_temp_min,
            h1_diasbp_max,
            h1_diasbp_min,
            h1_diasbp_noninvasive_max,
            h1_diasbp_noninvasive_min,
            h1_heartrate_max,
            h1_heartrate_min,
            h1_mbp_max,
            h1_mbp_min,
            h1_mbp_noninvasive_max,
            h1_mbp_noninvasive_min,
            h1_resprate_max,
            h1_resprate_min,
            h1_spo2_max,
            h1_spo2_min,
            h1_sysbp_max,
            h1_sysbp_min,
            h1_sysbp_noninvasive_max,
            h1_sysbp_noninvasive_min,
            h1_temp_max,
            h1_temp_min,
            d1_bun_max,
            d1_bun_min,
            d1_calcium_max,
            d1_calcium_min,
            d1_creatinine_max,
            d1_creatinine_min,
            d1_glucose_max,
            d1_glucose_min,
            d1_hco3_max,
            d1_hco3_min,
            d1_hemaglobin_max,
            d1_hemaglobin_min,
            d1_hematocrit_max,
            d1_hematocrit_min,
            d1_platelets_max,
            d1_platelets_min,
            d1_potassium_max,
            d1_potassium_min,
            d1_sodium_max,
            d1_sodium_min,
            d1_wbc_max,
            d1_wbc_min,
            apache_4a_hospital_death_prob,
            apache_4a_icu_death_prob,
            aids,
            cirrhosis,
            diabetes_mellitus,
            hepatic_failure,
            immunosuppression,
            leukemia,
            lymphoma,
            solid_tumor_with_metastasis,
            apache_3j_bodysystem,
            apache_2_bodysystem]
    if st.button('Servival Test Result'):
        diagnosis = diabetes_prediction(np.asarray(heh))
        
        
    st.success(diagnosis) 

loaded_model = pickle.load(open("trained.sav", 'rb'))  
data = pd.read_csv("mrk1.csv")
#0 = int(input("0 value between 0 and 49.9k"))
main()