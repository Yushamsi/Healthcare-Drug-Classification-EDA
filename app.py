from flask import Flask, request, render_template
import joblib
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
        # Extracting form data into a dictionary
        input_data = {key: request.form.get(key) for key in request.form.keys()}

        # Special handling for numerical fields to ensure correct type
        numerical_fields = ['Dexa_Freq_During_Rx', 'Count_Of_Risks']
        for field in numerical_fields:
            input_data[field] = float(input_data[field]) if input_data[field] else 0

        # Creating a DataFrame from the form data
        input_df = pd.DataFrame([input_data])

        print("Input DataFrame columns:", input_df.columns.tolist())

        # Check if all expected columns are present
        expected_columns = ['Gender',
                            'Race',
                            'Ethnicity',
                            'Region',
                            'Age_Bucket',
                            'Ntm_Speciality',
                            'Ntm_Specialist_Flag',
                            'Gluco_Record_Prior_Ntm',
                            'Gluco_Record_During_Rx',
                            'Dexa_Freq_During_Rx',
                            'Dexa_During_Rx',
                            'Frag_Frac_Prior_Ntm',
                            'Frag_Frac_During_Rx',
                            'Risk_Segment_Prior_Ntm',
                            'Tscore_Bucket_Prior_Ntm',
                            'Risk_Segment_During_Rx',
                            'Tscore_Bucket_During_Rx',
                            'Change_T_Score',
                            'Change_Risk_Segment',
                            'Adherent_Flag',
                            'Idn_Indicator',
                            'Injectable_Experience_During_Rx',
                            'Comorb_Encounter_For_Screening_For_Malignant_Neoplasms',
                            'Comorb_Encounter_For_Immunization',
                            'Comorb_Encntr_For_General_Exam_W_O_Complaint,_Susp_Or_Reprtd_Dx',
                            'Comorb_Vitamin_D_Deficiency',
                            'Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified',
                            'Comorb_Encntr_For_Oth_Sp_Exam_W_O_Complaint_Suspected_Or_Reprtd_Dx',
                            'Comorb_Long_Term_Current_Drug_Therapy',
                            'Comorb_Dorsalgia',
                            'Comorb_Personal_History_Of_Other_Diseases_And_Conditions',
                            'Comorb_Other_Disorders_Of_Bone_Density_And_Structure',
                            'Comorb_Disorders_of_lipoprotein_metabolism_and_other_lipidemias',
                            'Comorb_Osteoporosis_without_current_pathological_fracture',
                            'Comorb_Personal_history_of_malignant_neoplasm',
                            'Comorb_Gastro_esophageal_reflux_disease',
                            'Concom_Cholesterol_And_Triglyceride_Regulating_Preparations',
                            'Concom_Narcotics',
                            'Concom_Systemic_Corticosteroids_Plain',
                            'Concom_Anti_Depressants_And_Mood_Stabilisers',
                            'Concom_Fluoroquinolones',
                            'Concom_Cephalosporins',
                            'Concom_Macrolides_And_Similar_Types',
                            'Concom_Broad_Spectrum_Penicillins',
                            'Concom_Anaesthetics_General',
                            'Concom_Viral_Vaccines',
                            'Risk_Type_1_Insulin_Dependent_Diabetes',
                            'Risk_Osteogenesis_Imperfecta',
                            'Risk_Rheumatoid_Arthritis',
                            'Risk_Untreated_Chronic_Hyperthyroidism',
                            'Risk_Untreated_Chronic_Hypogonadism',
                            'Risk_Untreated_Early_Menopause',
                            'Risk_Patient_Parent_Fractured_Their_Hip',
                            'Risk_Smoking_Tobacco',
                            'Risk_Chronic_Malnutrition_Or_Malabsorption',
                            'Risk_Chronic_Liver_Disease',
                            'Risk_Family_History_Of_Osteoporosis',
                            'Risk_Low_Calcium_Intake',
                            'Risk_Vitamin_D_Insufficiency',
                            'Risk_Poor_Health_Frailty',
                            'Risk_Excessive_Thinness',
                            'Risk_Hysterectomy_Oophorectomy',
                            'Risk_Estrogen_Deficiency',
                            'Risk_Immobilization',
                            'Risk_Recurring_Falls',
                            'Count_Of_Risks',
                            'Ntm_Speciality_Restructured'] 
        
        # Add the list of columns you expect based on your model training

        missing_columns = set(expected_columns) - set(input_df.columns)
        if missing_columns:
            print("Missing columns:", missing_columns)

        print("Input data:", input_df.head())

        # Preprocess the input data using the loaded preprocessor
        processed_features = preprocessor.transform(input_df)

        # If possible, convert processed features to a DataFrame and print
        # Note: This step is just for debugging and can be removed later
        if hasattr(preprocessor, 'get_feature_names_out'):
            transformed_columns = preprocessor.get_feature_names_out()
            transformed_df = pd.DataFrame(processed_features, columns=transformed_columns)
            print("Transformed DataFrame:\n", transformed_df.head())
        else:
            print("Processed features shape:", processed_features.shape)

        # Predict using the preprocessed features
        prediction = model.predict(processed_features)

        # Convert prediction to a meaningful outcome
        output = 'Persistent' if prediction[0] == 1 else 'Non-Persistent'

        return render_template('index.html', prediction_text=f'Prediction: {output}')


    except Exception as e:
        # Log the error if any and return the error message on the webpage
        print(f"An error occurred: {str(e)}")
        return render_template('index.html', prediction_text=f'An error occurred: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)