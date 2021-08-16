# imports
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import configuration
from configuration import OBJECT_FOLDER as objectfolder

# import trained pipeline
trained_pipeline = joblib.load(os.path.join(objectfolder, "trained_pipeline.pkl"))

def get_prediction_df(input_data):
    
    # convert input data into data frame
    df = pd.DataFrame(input_data)

    # set time stamp as index
    df.set_index("Time", inplace=True)
    df = df.astype(np.float32)

    # save time stamps for traceability
    time_stamps = df.index.tolist()

    # create product IDs
    product_IDs = []
    for time_stamp in time_stamps:
        id_part1 = time_stamp[2:4]
        id_part2 = time_stamp[5:7]
        id_part3 = time_stamp[8:10]
        id_part4 = "{0:0=4d}".format(time_stamps.index(time_stamp))

        id_complete = str(id_part1) + str(id_part2) + str(id_part3) + "_" + str(id_part4)
        product_IDs.append(id_complete)

    
    # get prediction from pipeline
    predictions = trained_pipeline.predict(df)

    # save predictions with additional information
    df_results = pd.DataFrame(
    {'Time Stamp': time_stamps,
     'Prediction': predictions,
     'Product ID': product_IDs
    })

    # replace numerical values by human-readable ones
    df_results["Prediction"].replace(to_replace=-1, value="Pass", inplace=True)
    df_results["Prediction"].replace(to_replace=1, value="Fail", inplace=True)

    return df_results


def get_metrics_scores():
    
    # insert holdout data set with target variable
    data = pd.read_csv(configuration.HOLDOUT_DATA_FILE)

    # set Time as index
    data = data.set_index("Time", inplace=False)

    # save correct labels
    y = data["Pass/Fail"]

    # drop label as model requires unlabeled data
    X = data.drop('Pass/Fail', axis=1, inplace = False)

    # execute prediction
    y_pred = trained_pipeline.predict(X)

    # calculate accuracy, precision, recall and F1-Score
    scores = np.array([accuracy_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred), f1_score(y, y_pred)])

    # round values to 4 decimals
    scores_rounded = np.around(scores, decimals =4)

    # return rounded scores
    return scores_rounded


def get_version_number():

    # return current version of pipeline
    return configuration.VERSION_NUMBER


def test_prediction():
    
    # load validation data file into data frame
    filepath = configuration.VALIDATION_DATA_FILE
    df = pd.read_csv(filepath)

    # print predictions for manual check
    print(get_prediction_df(df))

# main method
if __name__ == '__main__':
    test_prediction()

