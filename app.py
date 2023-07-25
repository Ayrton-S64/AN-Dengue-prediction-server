from flask import Flask, request, jsonify
from IPython.display import display
from flask_cors import CORS
from keras.models import load_model
import pandas as pd
import numpy as  np
from sklearn.tree import DecisionTreeClassifier
from joblib import load


cases_prediction_model = load_model('./modelos/modelo_dengue_reducido.keras')


# Load the decision tree model from the saved file# Load the saved model
severity_assesment_model = load('./modelos/severity_diagnosis-decision_tree_model.pkl')

illness_regression_model = load('./modelos/enfermedades-regression-model.pkl')
illness_prognosis_encoder = load('./encoders/enfermedades-encoder.pkl')

app = Flask(__name__)

CORS(app, origins='*')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        predictions = cases_prediction_model.predict(data)
        
        response = {'predictions': predictions.tolist()}
        return jsonify(response)
    
    except Exception as e:
        print('Error: ',str(e))
        return jsonify({'error': str(e)}), 400


@app.route('/assess-severity', methods=['POST'])
def  diagnose():
    try:
        data = request.json

        arrdata = np.array([[float(data['duracion_fiebre']),
                float(data['hemoglobina']),
                float(data['hematocritos']),
                float(data['globulos_blancos']),
                float(data['plaquetas']),
                float(data['alt']),
                float(data['creatinina']), 
                1 if (int(data['duracion_fiebre']) > 6) else 0,
                float(data['genero']),
                float(data['cefalea']),
                float(data['mialgia']),
                float(data['dolor_abdominal']),
                float(data['erupciones']),
                float(data['vomitos']),
                float(data['dificultad_respirar']),
                float(data['hemorragia']),
                float(data['organomegalia'])
                ]])
        
        data_df = pd.DataFrame(arrdata, columns=['DURACION FIEBRE', 'HEMOGLOBINA', 'HEMATOCRITOS', 'GLOBULOS BLANCOS', 'PLAQUETAS', 'ALT', 'CREATININA', 'EDAD_menor a 6', 'GENERO_M', 'CEFALEA_SI', 'MIALGIA_SI', 'DOLOR ABDOMINAL_SI', 'ERUPCIONES_SI', 'VOMITOS_SI', 'DIFICULTAD AL RESPIRAR_SI', 'HEMORRAGIA_SI', 'ORGANOMEGALIA_SI']) 
        
        predictions = severity_assesment_model.predict(data_df)
        
        response = {'prediction': predictions[0]}

        return jsonify(response)
    
    except Exception as e:
        print('Error: ',str(e))
        return jsonify({'error': str(e)}), 400
    

@app.route('/illness-prediction', methods=['POST'])
def classify_illness():
    try:
        data = request.json

        arrData = np.array([[
            0,
            float(data.get('sintoma_fiebre',0)),
            float(data.get('sintoma_dolor_cabeza',0)),
            float(data.get('sintoma_sangrado_boca',0)),
            float(data.get('sintoma_sangrado_nasal',0)),
            float(data.get('sintoma_dolor_muscular',0)),
            float(data.get('sintoma_dolor_articulaciones',0)),
            float(data.get('sintoma_vomitos',0)),
            float(data.get('sintoma_erupcion_cutanea',0)),
            float(data.get('sintoma_diarrea',0)),
            float(data.get('sintoma_hipotension',0)),
            float(data.get('sintoma_derrame_pleural',0)),
            float(data.get('sintoma_ascitis',0)),
            float(data.get('sintoma_hemorragia_gastrointestinal',0)),
            float(data.get('sintoma_hinchazon',0)),
            float(data.get('sintoma_nauseas',0)),
            float(data.get('sintoma_escalofrios',0)),
            float(data.get('sintoma_mialgia',0)),
            float(data.get('sintoma_problemas_digestivos',0)),
            float(data.get('sintoma_fatiga',0)),
            float(data.get('sintoma_lesiones_piel',0)),
            float(data.get('sintoma_dolor_estomago',0)),
            float(data.get('sintoma_dolor_orbital',0)),
            float(data.get('sintoma_dolor_cuello',0)),
            float(data.get('sintoma_debilidad',0)),
            float(data.get('sintoma_dolor_espalda',0)),
            float(data.get('sintoma_perdida_peso',0)),
            float(data.get('sintoma_sangrado_encias',0)),
            float(data.get('sintoma_ictericia',0)),
            float(data.get('sintoma_coma',0)),
            float(data.get('sintoma_mareo',0)),
            float(data.get('sintoma_inflamacion',0)),
            float(data.get('sintoma_ojos_rojos',0)),
            float(data.get('sintoma_perdida_apetito',0)),
            float(data.get('sintoma_perdida_orina',0)),
            float(data.get('sintoma_ritmo_cardiaco_lento',0)),
            float(data.get('sintoma_dolor_abdominal',0)),
            float(data.get('sintoma_sensibilidad_luz',0)),
            float(data.get('sintoma_piel_amarilla',0)),
            float(data.get('sintoma_ojos_amarilla',0)),
            float(data.get('sintoma_distorsion_facial',0)),
            float(data.get('sintoma_microcefalia',0)),
            float(data.get('sintoma_rigidez',0)),
            float(data.get('sintoma_boca_amarga',0)),
            float(data.get('sintoma_convulsiones',0)),
            float(data.get('sintoma_anemia',0)),
            float(data.get('sintoma_orina_cola_cola',0)),
            float(data.get('sintoma_hipoglucemia',0)),
            float(data.get('sintoma_postraccion',0)),
            float(data.get('sintoma_fiebre_alta',0)),
            float(data.get('sintoma_rigidez_cuello',0)),
            float(data.get('sintoma_irritabilidad',0)),
            float(data.get('sintoma_confusion',0)),
            float(data.get('sintoma_temblores',0)),
            float(data.get('sintoma_paralisis',0)),
            float(data.get('sintoma_glangios_linfaticos_inflamados',0)),
            float(data.get('sintoma_respiracion_restringida',0)),
            float(data.get('sintoma_dolor_dedos_pie',0)),
            float(data.get('sintoma_dolor_dedos_manos',0)),
            float(data.get('sintoma_irritacion_labios',0)),
            float(data.get('sintoma_picazon',0)),
            float(data.get('sintoma_ulceras',0)),
            float(data.get('sintoma_perdidas_unias_pie',0)),
            float(data.get('sintoma_dificultad_hablar',0)),
            float(data.get('sintoma_eritema_multiforme',0))
        ]])

        columns = ['id','sudden_fever','headache','mouth_bleed','nose_bleed','muscle_pain','joint_pain','vomiting','rash','diarrhea','hypotension','pleural_effusion','ascites','gastro_bleeding','swelling','nausea','chills','myalgia','digestion_trouble','fatigue','skin_lesions','stomach_pain','orbital_pain','neck_pain','weakness','back_pain','weight_loss','gum_bleed','jaundice','coma','diziness','inflammation','red_eyes','loss_of_appetite','urination_loss','slow_heart_rate','abdominal_pain','light_sensitivity','yellow_skin','yellow_eyes','facial_distortion','microcephaly','rigor','bitter_tongue','convulsion','anemia','cocacola_urine','hypoglycemia','prostraction','hyperpyrexia','stiff_neck','irritability','confusion','tremor','paralysis','lymph_swells','breathing_restriction','toe_inflammation','finger_inflammation','lips_irritation','itchiness','ulcers','toenail_loss','speech_problem','bullseye_rash']

        data_df = pd.DataFrame(arrData, columns=columns) 

        probabilities = illness_regression_model.predict_proba(data_df)

        # Get the original prognosis names from the encoder
        prognosis_names = illness_prognosis_encoder.inverse_transform(illness_regression_model.classes_.reshape(-1, 1))
        prognosis_names = [str(name[0]) for name in prognosis_names]

        # Create a list of dictionaries, each containing prognosis names and their corresponding probabilities for each sample
        results = []
        for prob_sample in probabilities:
            result = {}
            for name, prob in zip(prognosis_names, prob_sample):
                result[name] = float(prob)
            results.append(result)


        response = {
            'prediction': results,
        }
        return jsonify(response)
    
    except Exception as e:
        print('Error: ',str(e))
        return jsonify({'error': str(e)}), 400

if __name__ ==  '__main__':
    app.run()