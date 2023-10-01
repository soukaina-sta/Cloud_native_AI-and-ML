from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

app = Flask(__name__)


gender_encoder = LabelEncoder()
ever_married_encoder = LabelEncoder()
work_type_encoder = LabelEncoder()
Residence_type_encoder = LabelEncoder()
smoking_status_encoder = LabelEncoder()


training_gender_data = ['Male', 'Female']
training_ever_married_data = ['Yes', 'No']
training_work_type_data = ['Private', 'Self-employed', 'Govt_job']
training_Residence_type_data = ['Urban', 'Rural']
training_smoking_status_data = ['formerly smoked', 'never smoked', 'smokes']

gender_encoder.fit(training_gender_data)
ever_married_encoder.fit(training_ever_married_data)
work_type_encoder.fit(training_work_type_data)
Residence_type_encoder.fit(training_Residence_type_data)
smoking_status_encoder.fit(training_smoking_status_data)

# Chargez le modèle après avoir ajusté les encodeurs de label
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.to_dict()

    features['gender'] = gender_encoder.transform([features['gender']])
    features['ever_married'] = ever_married_encoder.transform([features['ever_married']])
    features['work_type'] = work_type_encoder.transform([features['work_type']])
    features['Residence_type'] = Residence_type_encoder.transform([features['Residence_type']])
    features['smoking_status'] = smoking_status_encoder.transform([features['smoking_status']])

    features = list(features.values())
    features = list(map(float, features))

    final_features = np.array(features).reshape(1, 10)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Stroke prediction is :  {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
