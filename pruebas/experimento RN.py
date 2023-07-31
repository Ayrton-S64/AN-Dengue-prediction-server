from keras.models import load_model
import numpy as np

model = load_model('modelo_dengue_reducido.keras')

print('entrada: ',model.layers[0].input_shape)

model.summary()
 #	iq	sj	month	year	weekofyear	humidity_percent	avg_temp_c	max_temp_c	min_temp_c	precip_mm	quarter	day
data = np.array([[0, 1, 1, 2011, 52, 96.627143, 26.900000, 33.2, 22.0, 54.7, 1, 1],
                [0, 1, 1, 2011, 52, 76.627143, 32.900000, 36.2, 28.0, 7.7, 1, 1],
                [0, 1, 1, 2011, 52, 98.627143, 22.900000, 26.2, 18.0, 3.0, 1, 1]])

print('data: ',data.shape)

predictions = model.predict(data)

print(predictions)