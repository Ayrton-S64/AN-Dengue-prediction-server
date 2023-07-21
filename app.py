from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import pandas as pd
import numpy as  np
from sklearn.tree import DecisionTreeClassifier
from joblib import load


prediction_model = load_model('./modelos/modelo_dengue_reducido.keras')


# Load the decision tree model from the saved file# Load the saved model
loaded_model = load('./modelos/severity_diagnosis-decision_tree_model.pkl')

app = Flask(__name__)

CORS(app, origins='*')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    predictions = prediction_model.predict(data)
    
    response = {'predictions': predictions.tolist()}
    return jsonify(response)


@app.route('/assess-severity', methods=['POST'])
def  diagnose():
    # Make predictions using the loaded model
    data = request.json
    print('inputs-----------------------------')
    print(data)
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
    print('arreglo------------------------')
    print(arrdata)
    data_df = pd.DataFrame(arrdata, columns=['DURACION FIEBRE', 'HEMOGLOBINA', 'HEMATOCRITOS', 'GLOBULOS BLANCOS', 'PLAQUETAS', 'ALT', 'CREATININA', 'EDAD_menor a 6', 'GENERO_M', 'CEFALEA_SI', 'MIALGIA_SI', 'DOLOR ABDOMINAL_SI', 'ERUPCIONES_SI', 'VOMITOS_SI', 'DIFICULTAD AL RESPIRAR_SI', 'HEMORRAGIA_SI', 'ORGANOMEGALIA_SI']) 
    # new_data = pd.read_csv('new_data.csv')
    print('pandas--------------------------')
    print(data_df)
    # Preprocess the new_data as required
    predictions = loaded_model.predict(data_df)
    print('prediccion----------')
    print(predictions)
    response = {'prediction': predictions[0]}

    return jsonify(response)

    # Do something with the predictions

if __name__ ==  '__main__':
    app.run()