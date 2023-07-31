# from sklearn.externals import joblib
import joblib
import pandas as pd
import numpy as np

test_df = pd.read_csv('./Data/enfermedades/test.csv')
test_df.head()

model = joblib.load('./modelos/enfermedades-regression-model.pkl')
enc = joblib.load('./encoders/enfermedades-encoder.pkl')

# Make predictions using the loaded model and get the probabilities for all samples
probabilities = model.predict_proba(test_df)

# Get the original prognosis names from the encoder
prognosis_names = enc.inverse_transform(model.classes_.reshape(-1, 1))
prognosis_names = [str(name[0]) for name in prognosis_names]

# Create a list of dictionaries, each containing prognosis names and their corresponding probabilities for each sample
results = []
for prob_sample in probabilities:
    result = {}
    for name, prob in zip(prognosis_names, prob_sample):
        result[name] = float(prob)
    results.append(result)

print(results)



# # Make predictions
# predictions = model.predict_proba(test_df)

# # Get the sorted indices of predictions and take the top 3
# sorted_prediction_ids = np.argsort(-predictions, axis=1)
# top_3_prediction_ids = sorted_prediction_ids[:,:3]

# # Because enc.inverse_transform expects a specific shape (a 2D array with 1 column) we can save the original shape to reshape to after decoding
# original_shape = top_3_prediction_ids.shape
# top_3_predictions = enc.inverse_transform(top_3_prediction_ids.reshape(-1, 1))
# top_3_predictions = top_3_predictions.reshape(original_shape)
# print(top_3_predictions[:10]) # Spot check our first 10 values

# # Now to get our array of labels into a single column for our submission we can just join on on a space across axis 1
# test_df['prognosis'] = np.apply_along_axis(lambda x: np.array(' '.join(x), dtype="object"), 1, top_3_predictions)
# print(test_df['prognosis'][:10]) # Spot check our first 10 values