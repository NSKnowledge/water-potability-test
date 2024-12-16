import numpy as np
import pandas as pd
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

def load_data(filepath:str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error Loading data from {filepath}:{e}")

# test_data = pd.read_csv("./data/processed/test_processed.csv")

def prepare_data(data:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X_test = data.drop(columns=['Potability'],axis=1)
        y_test = data['Potability']
        return X_test,y_test
    except Exception as e:
        raise Exception(f"Error preparing data:{e}")
    
def load_model(filepath:str):
    try:
        with open(filepath,"rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model:{e}")

# model = pickle.load(open("model.pkl", "rb"))
# y_pred = model.predict(X_test)



def eval_score(model,X_test:pd.Series, y_test:pd.Series) -> dict:
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1score =f1_score(y_test,y_pred)

        metrics_dict ={
            "accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "f1score":f1score
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error making prediction:{e}")


def save_metrics(metrics:dict, filepath:str) ->None:
    try:
        with open(filepath,"w") as file:
            json.dump(metrics,file,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics:{e}")
    

def main():
    try:
        metrics_filepath ="reports/metrics.json"
        model_path ="models/model.pkl"
        processed_testdata_path =r"./data/processed/test_processed.csv"

        testdata_path = load_data(processed_testdata_path)
        X_test, y_test =prepare_data(testdata_path)
        model = load_model(model_path)

        metrics_dict = eval_score(model,X_test, y_test)
        save_metrics(metrics_dict,metrics_filepath)
        
    except Exception as e:
        raise Exception(f"An error occured :{e}")


if __name__=="__main__":
    main()