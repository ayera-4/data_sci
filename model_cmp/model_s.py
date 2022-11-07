#import
import pandas as pd

#import sklearn libraries
from sklearn.datasets import load_iris, load_boston, load_diabetes, load_digits, load_wine, load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#function to evaluate models on the different datasets
def evaluate_model(dataC, modelC, cls=True):
  #load datasetz
  X = dataC.data
  y = dataC.target

  #Create train/test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

  # initialize/ train/ test models
  model = modelC # load model
  t_model = model.fit(X_train, y_train) # train model
  y_pred = t_model.predict(X_test) # predict on new data

  # Evaluate model
  if cls == True:
    acc_scores = round(accuracy_score(y_test, y_pred), 3)
  else:
    acc_scores = "N/A"
  r2_scores = round(r2_score(y_test, y_pred), 3)

  return acc_scores, r2_scores


# function to check model performance
def get_model_results(data_list, classifier_list, regressor_list):
  results = {}
  #Loop through the datasets list of constructors
  for data_n, data in data_list.items():
    models = []
    accs = []
    r2s = []
    #Loop through the classifier list of constructors
    for model_n, model in classifier_list.items():
      a,b = evaluate_model(data, model, cls=True)
      models.append(model_n)
      accs.append(a)
      r2s.append(b)
    #Loop through the regressor list of constructors
    for model_n, model in regressor_list.items():
      a,b = evaluate_model(data, model, cls=False)
      models.append(model_n)
      accs.append(a)
      r2s.append(b)
    #save results into dictionary
    results[data_n] = {"Models": models, "Acc_scores":accs, "R2_scores":r2s}

  return results

#Experiment data structures
# list of dataset constructors
datasets = {
    "Iris":load_iris(), # "Boston":load_boston(),
    "Diabetes":load_diabetes(), "Digits":load_digits(),
    "Wine":load_wine(), "Breast_Cancer":load_breast_cancer()
}

#list of classifier model constructors
class_model = {
    "DecisionTreeClassifier":DecisionTreeClassifier(),
    "RandomForestClassifier":RandomForestClassifier(),
    "KNeighborsClassifier":KNeighborsClassifier(), 
    "MLPClassifier":MLPClassifier(),
    "GaussianNB":GaussianNB(), 
}

#list of regression model constructors
reg_model = {
    "DecisionTreeRegressor":DecisionTreeRegressor(),
    "RandomForestRegressor":RandomForestRegressor(),
    "KNeighborsRegressor":KNeighborsRegressor(),
    "MLPRegressor":MLPRegressor(),
    "linear_model.Ridge":linear_model.Ridge(),
    "linear_model.LinearRegression":linear_model.LinearRegression()
} 

#Run experiment and store results

# fetch model results
dict_data = get_model_results(datasets, class_model, reg_model)

#Create dataframe to store results
for result in dict_data:
  print(f"Models results on {result} dataset")
  df = pd.DataFrame().from_dict(dict_data[result]).set_index("Models")
  print(df)
  print("\n")
