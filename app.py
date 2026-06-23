import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# pastikan file ini ada di folder yang sama: model.pkl dan scaler.pkl
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f) # diasumsikan: [Decision Tree, SVC]
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except (FileNotFoundError, ModuleNotFoundError, pickle.PickleError) as e:
    raise RuntimeError(
        'Gagal memuat model atau scaler. Pastikan file model.pkl dan scaler.pkl ada, dan scikit-learn telah terpasang.'
    ) from e

model_names = ['Decision Tree', 'SVC']

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Pregnancies': int(request.form['pregnancies']),
        'Glucose': int(request.form['glucose']),
        'BloodPressure': int(request.form['blood_press']),
        'SkinThickness': int(request.form['skin_thickn']),
        'Insulin': int(request.form['insulin']),
        'BMI': float(request.form['bmi']),
        'DiabetesPedigreeFunction': float(request.form['diabetes_pedigree']),
        'Age': int(request.form['age'])
    }
    
    df = pd.DataFrame(data, index=[0])
    X = scaler.transform(df)
    
    clf = model[model_names.index(request.form['model'])]
    y = clf.predict(X)
    prediction = 'Diabetic' if int(y[0]) == 1 else 'Non-Diabetic'
    
    return render_template('index.html', prediction=prediction, model_names=model_names)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)