# imports
import os
import pandas as pd
import numpy as np
from math import sqrt
import joblib

# imports for sklearn functionality
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# imports for data balancing
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler

# ignore warnings
import warnings
warnings.simplefilter(action='ignore')

# import setting from configuration
import configuration
from configuration import OBJECT_FOLDER as objectfolder


def get_data():
    
    # load training data file
    filepath = configuration.TRAINING_DATA_FILE
    return pd.read_csv(filepath)


def adjust_columns(data):
    
    # drop duplicates
    data.drop_duplicates(inplace=True, subset=["Time"])

    # set time stamp as index
    data.set_index(keys=["Time"], inplace=True)

    # drop mostly empty columsn
    mostly_empty_columns=data.columns[data.isnull().mean()>0.5]
    data.drop(mostly_empty_columns, axis=1, inplace=True)

    #interpolate
    data.interpolate(inplace=True)
    data.fillna(method='bfill', inplace=True)

    # drop constant features
    isConstant = data.nunique() == 1
    constantColumns = data.columns[isConstant]
    data.drop(constantColumns, axis = 1, inplace=True)

    # export objects
    joblib.dump(mostly_empty_columns,   os.path.join(objectfolder, 'mostly_empty_columns.pkl'))
    joblib.dump(constantColumns,        os.path.join(objectfolder, 'constant_columns.pkl'))
  
    return data


def scale_data(data):
    
    # divide set into numeric and target
    data_numeric =  data[data.columns[data.columns != 'Pass/Fail']]
    data_target = data[data.columns[data.columns == 'Pass/Fail']]

    # create a scaler, train scaler and transform data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data_numeric)

    # create a new DataFrame with the standardized data and with the original labels
    data_scaled = pd.DataFrame(data = scaled, columns=data_numeric.columns)
    
    # put back the non numeric variable
    data_target.reset_index(inplace=True)
    data_scaled['Pass/Fail'] = data_target['Pass/Fail']

    # export trained scaler
    joblib.dump(scaler,   os.path.join(objectfolder, 'scaler.pkl'))
    
    return data_scaled


def reduce_dimension(data):
    
    # get the numerical data
    data_numeric = data[data.dtypes[data.dtypes == 'float64'].index]

    # Execute PCA so that 95% of variance are explained
    pca = PCA(.95, random_state=42)
    principal_components = pca.fit_transform(data_numeric)
    data_principal = pd.DataFrame(data = principal_components)

    # save output of PCA as array
    x = np.array(data_principal)
    y = np.array(data['Pass/Fail'])

    # features are selected via linear regression
    estimator = LinearRegression()
    rfe = RFE(estimator)
    selector = rfe.fit(x, y)

    # reduce data frame to only the selected variables
    selected_features = data_principal.columns[selector.support_]
    
    # reduce variables
    data_principal_reduced = data_principal[selected_features]
    data_principal_reduced["Pass/Fail"]=data["Pass/Fail"]

    # save PCA and RFE as objects for later
    joblib.dump(pca,   os.path.join(objectfolder, 'pca.pkl'))
    joblib.dump(rfe,   os.path.join(objectfolder, 'rfe.pkl'))    

    return data_principal_reduced


def undersample_data(data):

    # train test split
    train, test = train_test_split(data, test_size = 0.3, random_state=42)

    # separate data set into features and target
    X = data.loc[:, data.columns != 'Pass/Fail']
    y = data.loc[:, data.columns == 'Pass/Fail']

    # take majority class and reduce instances, the minority class is not changed
    rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)

    # execute resampling
    X_rus, y_rus = rus.fit_resample(X, y)

    #j oining features and target to one dataframe
    y_rus.columns = ['Pass/Fail']
    train_undersampled = X_rus.join(y_rus)
    train_undersampled = train_undersampled.sample(frac=1).reset_index(drop=True)

    # select randomly and scramble rows
    train_undersampled = train_undersampled.append(train.sample(frac=1)[0:500], sort=False)
    train_undersampled = train_undersampled.sample(frac=1).reset_index(drop=True)

    return train_undersampled


def fit_classifer(train):

    # divide train and test set into X and y each
    X_train = np.array(train.loc[:,train.columns !='Pass/Fail'])
    y_train = np.array(train.loc[:,train.columns =='Pass/Fail'])

    # create algorithm and train it
    classifier = RandomForestClassifier(n_estimators = 500, max_depth = 20, random_state = 42)
    classifier.fit(X_train, y_train.ravel())

    # export trained model
    joblib.dump(classifier,   os.path.join(objectfolder, 'trained_model.pkl'))
   

def execute_training():
    
    print("Training started.")

    # preprocessing steps
    data = reduce_dimension(scale_data(adjust_columns(get_data())))

    # undersampling of train set
    train = undersample_data(data)

    # fit model
    fit_classifer(train)
    
    # objetcs are saved and exported within functions

    print("Training finished.")

# main method
if __name__ == '__main__':
    execute_training()

