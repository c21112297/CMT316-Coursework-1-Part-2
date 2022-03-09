import pandas as pd
import math
import sklearn as sk
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

### Imports the data from the files using the read_csv command
trainingData = pd.read_csv(r"C:\Users\jacob\OneDrive\Desktop\real-state\train_full_Real-estate.csv")
testingData = pd.read_csv(r"C:\Users\jacob\OneDrive\Desktop\real-state\test_full_Real-estate.csv")

### Setting up the data in a dataframe 
trainingDataDF = pd.DataFrame(trainingData)
testingDataDF = pd.DataFrame(testingData)

#print(trainingDataDF.head())

### Checks if the values in the 'Y house price of unit area' column are
### greater than or equal to 30 and assigns 1 if they are and 0 otherwise
### by using .astype(Int) to make it a boolean series of True and False.
trainingDataDF['Y house price of unit area'] = (trainingDataDF['Y house price of unit area'] >= 30).astype(int)
testingDataDF['Y house price of unit area'] = (testingDataDF['Y house price of unit area'] >= 30).astype(int)

### These two lines shuffle the training and testing sets so the model
### doesnt overfit 
trainingDataDF = trainingDataDF.sample(frac = 1)
testingDataDF = testingDataDF.sample(frac = 1)

#print(trainingDataDF.head())

### Selects the appropriate columns and stores them
Ytrain = trainingDataDF.iloc[:,7]
Xtrain = trainingDataDF.iloc[:,:7]
#print(Ytrain)
#print(Xtrain)

Ytest = testingDataDF.iloc[:,7]
Xtest = testingDataDF.iloc[:,:7]
print(type(Ytest))
#print(Xtest)

### Classification - This is where the Prediction is made using SVM.
svm_model=sk.svm.SVC(kernel="linear",gamma='auto')
svm_model.fit(Xtrain, Ytrain)
classificationPrediction = svm_model.predict(Xtest)
accuracy = accuracy_score(Ytest, classificationPrediction)
print('Accuracy: '+ str(round(accuracy, 3)))

#print("SVM Classification:")
#print(svm_model.predict(Xtest))

### Regression - This is where the Prediction is made using LinearRegression
linearRegression_model=LinearRegression()
linearRegression_model.fit(Xtrain, Ytrain)
regressionPrediction = linearRegression_model.predict(Xtest)
mse = mean_squared_error(Ytest, regressionPrediction)
print('Root Means Squared Error: ' + str(math.sqrt(mse)))

#print("Linear Regression Classification:")
#print(linearRegression_model.predict(Xtest))


