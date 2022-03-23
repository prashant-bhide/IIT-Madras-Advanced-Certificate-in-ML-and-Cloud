# Using flask to make an api

# Importing necessary libraries and functions
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import pickle

# Creating a Flask app
app = Flask(__name__)

# Loading the KS Statistics table for gender prediction probability band ranges
df_KStable = pd.read_csv('df_KStable.csv')

# Extracting the min and max probability values from the top 3 and bottom 3 probability bands
top_3_band = df_KStable.loc[3]
class_1_min_prob = top_3_band.min_prob
bottom_3_band = df_KStable.loc[8]
class_0_max_prob = bottom_3_band.max_prob

# Loading the test dataset for gender and age class predictions
df_scn1_test = pd.read_csv('df_scn1_test.csv',
    dtype={'device_id':str, 'gender':'category', 'age':'category', 'train_test_flag':'category'})

# Loading the saved models for gender and age class predictions
scn1_gender_LR = pickle.load(open('scn1_gender_LR.pkl', 'rb'))
scn1_age_LR = pickle.load(open('scn1_age_LR.pkl', 'rb'))

# on the terminal type: curl http://127.0.0.1:5000/
@app.route('/', methods = ['GET'])
def home():
	if(request.method == 'GET'):
		data = 'Invoke the API with url as <URL>/predict_campaigns/<num>'
		return jsonify({'Usage': data})

# on the terminal type: curl http://127.0.0.1:5000/predict_campaigns/50
@app.route('/predict_campaigns/<int:num>', methods = ['GET'])
def predict_campaigns(num):
    if ((num < 1) or (num > 50)):
        result = {'Usage':'Invalid input '+str(num)}
        return jsonify(result)

    # Generate the test dataset with sample size as specified in the URL
    df_scn1_test_sample = df_scn1_test.sample(n=num)

    # Extract the X component from sample dataset
    df_scn1_Xtest = df_scn1_test_sample.drop(columns=['device_id','gender','age','train_test_flag'])

    # Generate the gender and age class predictions
    Yprob_gender = scn1_gender_LR.predict_proba(df_scn1_Xtest)
    Ypred_gender = scn1_gender_LR.predict(df_scn1_Xtest)
    Ypred_age = scn1_age_LR.predict(df_scn1_Xtest)
    #Yprobs_age = scn1_age_LR.predict_proba(df_scn1_Xtest)

    # Prepare the result list
    result_list = []
    for i in range(num):
        val = df_scn1_test_sample.iloc[i]
        gender_prob = round(Yprob_gender[i][1], 4)
        gender_pred = ('Female' if (gender_prob < class_0_max_prob) 
                        else 'Male' if (gender_prob > class_1_min_prob)
                        else '< Not in Top 3 / Bottom 3 Band >')
        row = {
            '0_device_id':val.device_id,
            #'1_gender_prob':float(gender_prob),
            #'1_gender_pred':('Female' if Ypred_gender[i]=='0' else 'Male'),
            '1_gender_pred':gender_pred,
            '2_age_group_pred':('[0-24]' if Ypred_age[i]=='0' else '[24-32]' if Ypred_age[i]=='1' else '[32+]'),
            'Campaign 1':('YES' if gender_pred=='Female' else '---'),
            'Campaign 2':('YES' if gender_pred=='Female' else '---'),
            'Campaign 3':('YES' if gender_pred=='Male' else '---'),
            'Campaign 4':('YES' if Ypred_age[i]=='0' else '---'),
            'Campaign 5':('YES' if Ypred_age[i]=='1' else '---'),
            'Campaign 6':('YES' if Ypred_age[i]=='2' else '---'),
        }
        result_list.append(row)
    
    # Return the result list as json
    return jsonify(result_list)

# driver function
if __name__ == '__main__':
    #app.run()
	#app.run(debug = True)
    app.run(debug=True, host="0.0.0.0")
