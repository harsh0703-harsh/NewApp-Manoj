from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
import sklearn


model = pickle.load(open('regression_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')



@app.route("/predict",methods=['POST'])
def predict():
    if request.method == "POST":
        tv = int(request.form['TV'])
        newspaper=float(request.form['newspaper'])
        radio=int(request.form['radio'])

        g = scaler.transform([[tv,newspaper,radio]])
        prediction = model.predict(g)
        return render_template("index.html",text = prediction)





if __name__=="__main__":
    app.run(debug=True)