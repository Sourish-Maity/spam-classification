from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))  # Assuming you saved TF-IDF vectorizer during training


@app.route('/')
def index():
    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    input_mail = request.form['input_mail']

    # Transform input mail into features using the pre-trained vectorizer
    input_data_features = vectorizer.transform([input_mail])

    # Make prediction using the loaded model
    prediction = model.predict(input_data_features)[0]

    # Map prediction to text
    if prediction == 1:
        result = 'This is not spam mail'
    else:
        result = 'Spam mail'

    return render_template('index.html', prediction=result, input_mail=input_mail)


if __name__ == '__main__':
    app.run(debug=True)