# Impport imp/required libraries
import numpy as np
import pandas as pd
import statistics
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Reading data
df = pd.read_csv("Training.csv")

# Dropping last column because of no data in last column
df = df.dropna(axis = 1)
 
# Converting prognosis column into integer
Lab_encoder = LabelEncoder()
df["prognosis"] = Lab_encoder.fit_transform(df["prognosis"])

# Splitting data for Training and Testing
X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(
  X, y, test_size = 0.2, random_state = 42)

# Modelling
# Scoring metric for cross validation
def cross_valid_score(projector, X, y):
    return accuracy_score(y, projector.predict(X))
 
# Model initializing
classifiers = {
    "Support Vector Classifier":SVC(),
    "Gaussian Classifier":GaussianNB(),
    "Random Forest Classifier":RandomForestClassifier(random_state=24)
}
 
# Producing cross validation score for the models
for classifier in classifiers:
    model = classifiers[classfier]
    check_val = cross_val_score(model, X, y, cv = 10, 
                             n_jobs = -1, 
                             scoring = cross_valid_score)
    print(\n\n)
    print(classifier:)
    print(f"Probability: {np.mean(check_val)}")

# Checking accuracy on Train and Test data
svm_model = SVC() # Support Vector Machine Classifier
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

nb_model = GaussianNB() # Naive Bayes Classifier
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

rf_model = RandomForestClassifier(random_state=24) # Random Forest Classifier
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
 
print(f"Accuracy on train data by SVM Classifier\
: {accuracy_score(y_train, svm_model.predict(X_train))*100}")
 
print(f"Accuracy on test data by SVM Classifier\
: {accuracy_score(y_test, svm_preds)*100}")

print(f"Accuracy on train data by Naive Bayes Classifier\
: {accuracy_score(y_train, nb_model.predict(X_train))*100}")
 
print(f"Accuracy on test data by Naive Bayes Classifier\
: {accuracy_score(y_test, nb_preds)*100}")

print(f"Accuracy on train data by Random Forest Classifier\
: {accuracy_score(y_train, rf_model.predict(X_train))*100}")
 
print(f"Accuracy on test data by Random Forest Classifier\
: {accuracy_score(y_test, rf_preds)*100}")

# Fitting the model
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Reading test dataset
data = pd.read_csv("Testing.csv")
data = data.dropna(axis=1)
 
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])
 
# Prediction from each classifier
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)
 
final_preds = [mode([i,j,k])[0] for i,j,
               k in zip(svm_preds, nb_preds, rf_preds)]
 
print(f"Model Accuracy\
: {accuracy_score(test_Y, final_preds)*100}")

# symptoms dictionary
symptoms = X.columns.values

# Giving Input of symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index
    
# Taking symptoms, indexes of symptoms and predictions into a dictionary variable
data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}

# Defining a function for predictions
def predictDisease(symptoms):
   # Separating symptoms input using comma
    symptoms = symptoms.split(",")
     
    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
         
    # reshaping the input data which suits format of model
    input_data = pd.DataFrame(np.array(input_data).reshape(1,-1), columns = X.columns.values)
     
    # Predictions of each classifier
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
     
    # Final prediction
    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction":final_prediction
        }
    return predictions

