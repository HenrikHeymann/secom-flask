# imports
import pandas as pd
import os
import joblib
from datetime import datetime
from flask import Flask, request, redirect, url_for, render_template

# import of functionality within the application
import configuration
from ml_model import predict

# definition of the app
app = Flask(__name__)

# standard endpoint
@app.route('/', methods=['GET'])
def home():
   
    # by accessing the endpoint a GET request is triggered
    if request.method == 'GET':
        
        # index.html file is returned and displayed
        return render_template("index.html")
    
# prediction endpoint
@app.route('/prediction', methods=['GET', 'POST'])
def get_prediction():
   
    if request.method == 'GET':
        
        # files of production data are listed
        files = os.listdir(configuration.PRODUCTION_DATA_FOLDER)
        files.sort()

        # list of files is passed to prediction_input.html
        # html file is returned and displayed
        return render_template("prediction_input.html", list_of_files = files)
    
    # by pressing the submit button a POST request is made
    if request.method == 'POST':
        
        # with a POST request the predictions are triggered
        text = "Time and hour of prediction: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        # get the selected option from the dropdown menu
        selected_file = request.form.get("dropdown")

        # check if an option was selected
        if selected_file != '':
            
            # create empty dataframe
            df = pd.DataFrame()

            # build path of file
            filepath = os.path.join(configuration.PRODUCTION_DATA_FOLDER, selected_file)
            
            # load input data from selected file
            input_data = pd.read_csv(filepath)

            # fill dataframe with predictions from model
            df = predict.get_prediction_df(input_data)

        # display prediction_output.html to show the predictions as a table
        return render_template("prediction_output.html", pred_to_print = text, table=df.to_html(index = False, header=True, table_id="result_table"))

# monitoring endpoint
@app.route('/monitoring', methods=['GET'])
def get_evaluation():
   
    if request.method == 'GET':
        
        # get version number of model
        version_number = predict.get_version_number()

        # get evaluation metrics scores for model
        scores = predict.get_metrics_scores()

        # display results
        return render_template("monitoring.html", ver=version_number, acc=scores[0], pre=scores[1], rec=scores[2], f1=scores[3])

# main method
if __name__ == '__main__':
    #app.run(debug=True, host = '0.0.0.0')
    app.run(debug=True)
