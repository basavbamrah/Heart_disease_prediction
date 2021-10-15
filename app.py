#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from numpy.core.numeric import outer

# Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# default page of our web-app


@app.route('/')
def home():
    return render_template('cover.html')

# To use the predict button in our web-app


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    output = prediction[0]
    print(prediction)
    if(output == 1):
        return render_template('cover.html', prediction_text='Predicted :{} \n  !!! Do Consult a Doctor'.format(output))
    else:
        return render_template('cover.html', prediction_text='Predicted :{}\n  You Have A Healthy Heart'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
