from flask import Flask, request, jsonify
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
        return jsonify({'error': str(e)}), 400
    

@app.route('/illness-prediction', methods=['POST'])
def classify_illness():
    try:
        data = request.json
        probabilities = illness_regression_model.predict_proba(data)

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

        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ ==  '__main__':
    app.run()