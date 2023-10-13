from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline

# Initialize Flask Web Application

app = Flask(__name__)

@app.route('/',methods=['GET'])
def homePage():
    return render_template('index.html')

@app.route('/train',methods=['GET'])  #route to train the pipeline
def training():
    os.system('python main.py')
    return 'training successfully completed'

@app.route('/predict',methods=['POST','GET'])  # take inputs from user and predict
def index():
    if request.method == 'POST':
        try:
            credit_duration = int(request.form['credit_duration(month)'])
            credit_amount = float(request.form['credit_amount'])
            duration_of_current_employment: int(request.form['duration_of_current_employment'])
            installment_percent = float(request.form['installment_percent'])
            guarantors = int(request.form['guarantors'])
            duration_in_current_address = int(request.form['duration_in_current_address'])
            age= int(request.form['age'])
            concurrent_credits = float(request.form['concurrent_credits'])
            no_of_credits_at_the_bank = int(request.form['no_of_credits_at_the_bank'])
            no_of_dependents = int(request.form['no_of_dependents'])
            account_type = object(request.form['account_type'])
            payment_status_of_previous_loan = object(request.form['payment_status_of_previous_loan'])
            loan_purpose = object(request.form['loan_purpose'])
            savings_type = object(request.form['savings_type'])
            marital_status = object(request.form['marital_status'])
            most_valuable_asset = object(request.form['most_valuable_asset'])
            type_of_apartment = object(request.form['type_of_apartment'])
            occupation = object(request.form['occupation'])
            telephone = object(request.form['telephone'])
            foreign_worker = object(request.form['foreign_worker'])
            



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
