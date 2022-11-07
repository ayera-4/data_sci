# import libraries
import pandas as pd
import xgboost as xgb

from sklearn.datasets import load_iris, load_boston, load_diabetes, load_wine, load_digits, load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

def model_prep(dataC, modelC, cls=True):
  #load data from dataset
  X = dataC.data
  y = dataC.target

  #create train test split of data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

  #load train test model
  model = modelC #load
  t_model = model.fit(X_train, y_train) #train
  y_pred = t_model.predict(X_test) # predict on test set

  #Evaluate model
  if cls:
    acc_score = round(accuracy_score(y_test, y_pred), 3)
  else:
    acc_score = "n/a"
  
  r2s_score = round(r2_score(y_test, y_pred), 3)

  return acc_score, r2s_score

#create dictionary for dataset constructors
dataset = {
    "Iris": load_iris(),
    "Diabetes": load_diabetes(),
    "Wine": load_wine(),
    "Digits": load_digits(),
    "Breast_cancer": load_breast_cancer()
}

#Initialize dictionary to save results
results = {}

#loop through data for datasets
for data_n, data in dataset.items():
  accs = []
  r2s = []
  models = []
  #run classifier on data
  a, b = model_prep(data, xgb.XGBClassifier(), cls=True)
  models.append("xgbClassifier")
  accs.append(a)
  r2s.append(b)

  #run regressor on data
  a, b = model_prep(data, xgb.XGBRegressor(), cls=False)
  models.append("xgbRegressor")
  accs.append(a)
  r2s.append(b)

  results[data_n] = {"Models": models, "Acc_scores":accs, "R2_scores":r2s}
  
  # store results into dataframe
for result in results:
  print(f"Model results from {result} dataset")
  df = pd.DataFrame().from_dict(results[result]).set_index("Models")
  print(df)
  print("\n")
