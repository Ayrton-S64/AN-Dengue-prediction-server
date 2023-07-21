from joblib import load
import numpy as np
import pandas as pd

decision_tree = load('./modelos/severity_diagnosis-decision_tree_model.pkl')


#  ['DURACION FIEBRE', 
#   ' HEMOGLOBINA', 
#   'HEMATOCRITOS', 
#   'GLOBULOS BLANCOS',    
#   'PLAQUETAS', 
#   'ALT', 
#   'CREATININA', 
#   'EDAD_menor a 6',
#   'GENERO_M', 
#   'CEFALEA_SI', 
#   'MIALGIA_SI', 
#   'DOLOR ABDOMINAL_SI',
#   'ERUPCIONES_SI', 
#   'VOMITOS_SI', 
#   'DIFICULTAD AL RESPIRAR_SI',
#   'HEMORRAGIA_SI', 
#   'ORGANOMEGALIA_SI']
data = np.array([[3, 13, 38.8, 10.3, 190, 13, 0.6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
data_df = pd.DataFrame(data, columns=['DURACION FIEBRE', 'HEMOGLOBINA', 'HEMATOCRITOS', 'GLOBULOS BLANCOS', 'PLAQUETAS', 'ALT', 'CREATININA', 'EDAD_menor a 6', 'GENERO_M', 'CEFALEA_SI', 'MIALGIA_SI', 'DOLOR ABDOMINAL_SI', 'ERUPCIONES_SI', 'VOMITOS_SI', 'DIFICULTAD AL RESPIRAR_SI', 'HEMORRAGIA_SI', 'ORGANOMEGALIA_SI'])


print('data: ', data_df)

predictions = decision_tree.predict(data_df)

print('prediccion: ', predictions)