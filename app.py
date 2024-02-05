from flask import Flask, render_template, request
import os
import numpy as np
import joblib
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline
from mlProject.utils.common import feature_processor, numpy_to_pandas, ordinal_category_encode
from mlProject import logger
from pathlib import Path
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
            duration_of_current_employment = int(request.form['duration_of_current_employment'])
            installment_percent = float(request.form['installment_percent'])
            guarantors = int(request.form['guarantors'])
            duration_in_current_address = int(request.form['duration_in_current_address'])
            age = int(request.form['age'])
            concurrent_credits = float(request.form['concurrent_credits'])
            no_of_credits_at_the_bank = int(request.form['no_of_credits_at_the_bank'])
            no_of_dependents = int(request.form['no_of_dependents'])
            account_type = str(request.form['account_type'])
            payment_status_of_previous_loan = str(request.form['payment_status_of_previous_loan'])
            loan_purpose = str(request.form['loan_purpose'])
            savings_type = str(request.form['savings_type'])
            marital_status = str(request.form['marital_status'])
            most_valuable_asset = str(request.form['most_valuable_asset'])
            type_of_apartment = str(request.form['type_of_apartment'])
            occupation = str(request.form['occupation'])
            telephone = str(request.form['telephone'])
            foreign_worker = str(request.form['foreign_worker'])

            data = [credit_duration,credit_amount,duration_of_current_employment,
                    installment_percent,guarantors,duration_in_current_address,
                    age,concurrent_credits,no_of_credits_at_the_bank,no_of_dependents,
                    account_type,payment_status_of_previous_loan,loan_purpose,savings_type,
                    marital_status,most_valuable_asset,type_of_apartment,occupation,telephone,foreign_worker]
            
            reshaped_data = np.array(data).reshape(1,20)
            features = numpy_to_pandas(reshaped_data)
            transformed_data = ordinal_category_encode(features)
    
            # model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

            obj = PredictionPipeline()

            predict = obj.predict(transformed_data)

            if predict[0] == 1:
                result = 'Congratulations, you are eligible for a loan'
            else:
                result = 'Sorry, you are ineligible for a loan'

            return render_template('results.html', prediction = result)
        
        except Exception as e:
            logger.exception(e)
            print(f'The Exception message is: {e}')
            return 'something is wrong'
            #raise e
            
       # except Exception as e:
            
            

    else:
        return render_template('index.html')
    



if __name__ == '__main__':
    app.run(host='0.0.0.0') # Use this when deploying to AWS
    # app.run(host='0.0.0.0', port=8080, debug=True)
