# imports
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from configuration import OBJECT_FOLDER as objectfolder

# import auxiliary methods necessary for pipeline
from ml_model import preprocessors as pp

# load objects from training
mostly_empty_columns =  joblib.load(filename=os.path.join(objectfolder, "mostly_empty_columns.pkl"))
constant_columns =      joblib.load(filename=os.path.join(objectfolder, "constant_columns.pkl"))
scaler_imported =       joblib.load(filename=os.path.join(objectfolder, "scaler.pkl"))
pca_imported =          joblib.load(filename=os.path.join(objectfolder, "pca.pkl"))
rfe_imported =          joblib.load(filename=os.path.join(objectfolder, "rfe.pkl"))
model_imported =        joblib.load(filename=os.path.join(objectfolder, "trained_model.pkl"))

# define steps of pipeline
pipeline = Pipeline(
    [
        ('remove_mostly_empty_columns', pp.RemoveMostlyEmptyColumns(variables_to_drop=mostly_empty_columns)),

        ('interpolate_missing_values', pp.InterpolateMissingValues()),

        ('remove_constant_features', pp.RemoveConstantFeatures(variables_to_drop=constant_columns)),

        ('Standard_Scaler', scaler_imported),

        ('PCA', pca_imported),

        ('RFE', rfe_imported),

        ('Random_Forest', model_imported)
    ]
)

# export pipeline
def dump_pipeline():
    
    joblib.dump(pipeline, filename=os.path.join(objectfolder, "trained_pipeline.pkl"))

if __name__ == '__main__':
    print("Buiding process started.")
    dump_pipeline()
    print("Buiding process successful.")