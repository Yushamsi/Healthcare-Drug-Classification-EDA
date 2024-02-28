from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd 

app = Flask(__name__)

# Load the model and transformer
model = joblib.load('logistic_regression_model.pkl')
preprocessor = joblib.load('transformer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting specific form data
        gender = request.form['Gender']
        race = request.form['Race']
        ethnicity = request.form['Ethnicity']
        region = request.form['Region']
        age_bucket = request.form['Age_Bucket']
        ntm_speciality = request.form['Ntm_Speciality']
        ntm_specialist_flag = request.form['Ntm_Specialist_Flag']
        gluco_record_prior_ntm = request.form['Gluco_Record_Prior_Ntm']
        gluco_record_during_rx = request.form['Gluco_Record_During_Rx']
        dexa_freq_during_rx = request.form['Dexa_Freq_During_Rx']
        dexa_during_rx = request.form['Dexa_During_Rx']
        frag_frac_prior_ntm = request.form['Frag_Frac_Prior_Ntm']
        frag_frac_during_rx = request.form['Frag_Frac_During_Rx']
        risk_segment_prior_ntm = request.form['Risk_Segment_Prior_Ntm']
        tscore_bucket_prior_ntm = request.form['Tscore_Bucket_Prior_Ntm']
        risk_segment_during_rx = request.form['Risk_Segment_During_Rx']
        tscore_bucket_during_rx = request.form['Tscore_Bucket_During_Rx']
        change_t_score = request.form['Change_T_Score']
        change_risk_segment = request.form['Change_Risk_Segment']
        adherent_flag = request.form['Adherent_Flag']
        idn_indicator = request.form['Idn_Indicator']
        injectable_experience_during_rx = request.form['Injectable_Experience_During_Rx']
        comorb_encounter_for_screening_for_malignant_neoplasms = request.form['Comorb_Encounter_For_Screening_For_Malignant_Neoplasms']
        comorb_encounter_for_immunization = request.form['Comorb_Encounter_For_Immunization']
        comorb_encntr_for_general_exam_wo_complaint_susp_or_reprtd_dx = request.form['Comorb_Encntr_For_General_Exam_W_O_Complaint,_Susp_Or_Reprtd_Dx']
        comorb_vitamin_d_deficiency = request.form['Comorb_Vitamin_D_Deficiency']
        comorb_other_joint_disorder_not_elsewhere_classified = request.form['Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified']
        comorb_encntr_for_oth_sp_exam_wo_complaint_suspected_or_reprtd_dx = request.form['Comorb_Encntr_For_Oth_Sp_Exam_W_O_Complaint_Suspected_Or_Reprtd_Dx']
        comorb_long_term_current_drug_therapy = request.form['Comorb_Long_Term_Current_Drug_Therapy']
        comorb_dorsalgia = request.form['Comorb_Dorsalgia']
        comorb_personal_history_of_other_diseases_and_conditions = request.form['Comorb_Personal_History_Of_Other_Diseases_And_Conditions']
        comorb_other_disorders_of_bone_density_and_structure = request.form['Comorb_Other_Disorders_Of_Bone_Density_And_Structure']
        comorb_disorders_of_lipoprotein_metabolism_and_other_lipidemias = request.form['Comorb_Disorders_of_lipoprotein_metabolism_and_other_lipidemias']
        comorb_osteoporosis_without_current_pathological_fracture = request.form['Comorb_Osteoporosis_without_current_pathological_fracture']
        comorb_personal_history_of_malignant_neoplasm = request.form['Comorb_Personal_history_of_malignant_neoplasm']
        comorb_gastro_esophageal_reflux_disease = request.form['Comorb_Gastro_esophageal_reflux_disease']
        concom_cholesterol_and_triglyceride_regulating_preparations = request.form['Concom_Cholesterol_And_Triglyceride_Regulating_Preparations']
        concom_narcotics = request.form['Concom_Narcotics']
        concom_systemic_corticosteroids_plain = request.form['Concom_Systemic_Corticosteroids_Plain']
        concom_anti_depressants_and_mood_stabilisers = request.form['Concom_Anti_Depressants_And_Mood_Stabilisers']
        concom_fluoroquinolones = request.form['Concom_Fluoroquinolones']
        concom_cephalosporins = request.form['Concom_Cephalosporins']
        concom_macrolides_and_similar_types = request.form['Concom_Macrolides_And_Similar_Types']
        concom_broad_spectrum_penicillins = request.form['Concom_Broad_Spectrum_Penicillins']
        concom_anaesthetics_general = request.form['Concom_Anaesthetics_General']
        concom_viral_vaccines = request.form['Concom_Viral_Vaccines']
        risk_type_1_insulin_dependent_diabetes = request.form['Risk_Type_1_Insulin_Dependent_Diabetes']
        risk_osteogenesis_imperfecta = request.form['Risk_Osteogenesis_Imperfecta']
        risk_rheumatoid_arthritis = request.form['Risk_Rheumatoid_Arthritis']
        risk_untreated_chronic_hyperthyroidism = request.form['Risk_Untreated_Chronic_Hyperthyroidism']
        risk_untreated_chronic_hypogonadism = request.form['Risk_Untreated_Chronic_Hypogonadism']
        risk_untreated_early_menopause = request.form['Risk_Untreated_Early_Menopause']
        risk_patient_parent_fractured_their_hip = request.form['Risk_Patient_Parent_Fractured_Their_Hip']
        risk_smoking_tobacco = request.form['Risk_Smoking_Tobacco']
        risk_chronic_malnutrition_or_malabsorption = request.form['Risk_Chronic_Malnutrition_Or_Malabsorption']
        risk_chronic_liver_disease = request.form['Risk_Chronic_Liver_Disease']
        risk_family_history_of_osteoporosis = request.form['Risk_Family_History_Of_Osteoporosis']
        risk_low_calcium_intake = request.form['Risk_Low_Calcium_Intake']
        risk_vitamin_d_insufficiency = request.form['Risk_Vitamin_D_Insufficiency']
        risk_poor_health_frailty = request.form['Risk_Poor_Health_Frailty']
        risk_excessive_thinness = request.form['Risk_Excessive_Thinness']
        risk_hysterectomy_oophorectomy = request.form['Risk_Hysterectomy_Oophorectomy']
        risk_estrogen_deficiency = request.form['Risk_Estrogen_Deficiency']
        risk_immobilization = request.form['Risk_Immobilization']
        risk_recurring_falls = request.form['Risk_Recurring_Falls']
        count_of_risks = request.form['Count_Of_Risks']
        ntm_speciality_restructured = request.form['Ntm_Speciality_Restructured']

        # Combine all extracted values into a list
        input_features = [gender, race, ethnicity, region, age_bucket, ntm_speciality]
        
        # Assuming you have additional form fields to include
        additional_features = [request.form[field_name] for field_name in additional_field_names]
        input_features.extend(additional_features)

        # Create a DataFrame with the correct column names
        input_dataframe = pd.DataFrame([input_features], columns=categorical_columns + numerical_columns)
        
        # Preprocess the input data
        processed_features = preprocessor.transform(input_dataframe)
        
        # Predict
        prediction = model.predict(processed_features)
        
        # Assuming a binary classification model
        output = 'Positive' if prediction[0] == 1 else 'Negative'

        return render_template('index.html', prediction_text='Prediction: {}'.format(output))

    except Exception as e:
        # Handle any errors that occur during form data processing or prediction
        print(e)  # It's often helpful to log the exact error
        return render_template('index.html', prediction_text='An error occurred during prediction.')

if __name__ == "__main__":
    app.run(debug=True)
