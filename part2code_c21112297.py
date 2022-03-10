# Author: Jacob Back
# Version 2.0

'''
Import all the packages that are going to be used throughout the program
'''
import pandas as pd
import os
import nltk
import sklearn
from nltk.tokenize import word_tokenize
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

'''
Vectorisers are declared
'''
tfidf_vectoriser = TfidfVectorizer()
count_vectoriser = CountVectorizer()

'''
Paths where each categories articles are stored
'''
bbcBusinessFile = r"C:\Users\jacob\OneDrive\Desktop\CMT Assignment Part 2\bbc\business"
bbcEntertainmentFile = (r"C:\Users\jacob\OneDrive\Desktop\CMT Assignment Part 2\bbc\entertainment")
bbcPoliticsFile = (r"C:\Users\jacob\OneDrive\Desktop\CMT Assignment Part 2\bbc\politics")
bbcSportFile = (r"C:\Users\jacob\OneDrive\Desktop\CMT Assignment Part 2\bbc\sport")
bbcTechFile = (r"C:\Users\jacob\OneDrive\Desktop\CMT Assignment Part 2\bbc\tech")

'''
Lists where the articles are going to be appended to 
'''
business_list = []
entertainment_list = []
politics_list = []
sport_list = []
tech_list = []

print('Data is being gathered...\n')

'''
These five for loops are where the .txt files are gathered, lightly formatted
and added to a DataFrame 
'''
###BUSINESS###
for file in os.listdir(bbcBusinessFile):
    without_new_line = ''
    with open(bbcBusinessFile + '\\' +file) as j:
        lines = j.readlines()
        for line in lines:
            new_line = line.replace('\n', ' ')
            without_new_line += new_line
        business_list.append(without_new_line)
        bsdf = pd.DataFrame({'category': 'Business', 'articles': business_list})
        lines=''
        without_new_line=''
#print("Buisiness: " + str(len(bsdf)))
#print(bsdf)
#print(business_list)

###ENTERTAINMENT###
for file in os.listdir(bbcEntertainmentFile):
    without_new_line = ''
    with open(bbcEntertainmentFile + '\\' +file) as j:
        lines = j.readlines()
        for line in lines:
            new_line = line.replace('\n', ' ')
            without_new_line += new_line
        entertainment_list.append(without_new_line)
        entdf = pd.DataFrame({'category': 'Entertainment', 'articles': entertainment_list})
        lines=''
        without_new_line=''
#print("Entertainment: " + str(len(entertainment_list)))
#print(entdf)
#print(entertainment_list)

###POLITICS###
for file in os.listdir(bbcPoliticsFile):
    without_new_line = ''
    with open(bbcPoliticsFile + '\\' +file) as j:
        lines = j.readlines()
        for line in lines:
            new_line = line.replace('\n', ' ')
            without_new_line += new_line
        politics_list.append(without_new_line)
        poldf = pd.DataFrame({'category': 'Politics', 'articles': politics_list})
        lines=''
        without_new_line = ''
#print("Politics: " + str(len(politics_list)))
#print(poldf)
#print(politics_list)

###SPORT###
for file in os.listdir(bbcSportFile):
    without_new_line = ''
    with open(bbcSportFile + '\\' +file) as j:
        lines = j.readlines()
        for line in lines:
            new_line = line.replace('\n', ' ')
            without_new_line += new_line
        sport_list.append(without_new_line)
        sptdf = pd.DataFrame({'category': 'Sport', 'articles': sport_list})
        lines=''
        without_new_line = ''
#print("Sport: " + str(len(sport_list)))
#print(sptdf)
#print(sport_list)

###TECH###
for file in os.listdir(bbcTechFile):
    without_new_line = ''
    with open(bbcTechFile + '\\' +file) as j:
        lines = j.readlines()
        for line in lines:
            new_line = line.replace('\n', ' ')
            without_new_line += new_line
        tech_list.append(without_new_line)
        tchdf = pd.DataFrame({'category': 'Tech', 'articles': tech_list})
        lines=''
        without_new_line=''
#print("Tech: " + str(len(tech_list)))
#print(tchdf)
#print(tech_list)

'''
All DataFrames are combined into one central DataFrame 
'''
combined_df = pd.concat([bsdf, entdf, poldf, sptdf, tchdf])
print('Data has been collected!\n')
combined_df = combined_df.sample(frac = 1)
#print(combined_df)

business_list = []
entertainment_list = []
politics_list = []
sport_list = []
tech_list = []

'''
Data gathered from the DataFrame is split into Xtrain, Ytrain, Xtest, and
Ytest datasets with a ratio of 80% training and 20% testing
'''
Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(combined_df['articles'], combined_df['category'], random_state=0, test_size=0.2)

'''
The svm model is set ready to be used
'''
svm_model=sklearn.svm.SVC(kernel="linear",gamma='auto')

'''
For each feature the vectorised data is fit, a prediction is made, the
accuracy is calculated, the classification report is generated, cross
validation score for the training and testing sets is generated, and the
confusion matrix is generated.
''' 
### COUNT VECTORIZATION AS FEATURE###
count_vectoriser.fit(combined_df['articles'])
xtrain_count = count_vectoriser.transform(Xtrain)
xtest_count = count_vectoriser.transform(Xtest)

print('Count model is being created...\n')
svm_model.fit(xtrain_count, Ytrain)
print('Count model has been created!\n')

print('Predicting using count...\n')
count_svm_predictions = svm_model.predict(xtest_count)

percentage_accuracy = accuracy_score(Ytest, count_svm_predictions)
raw_accuracy = accuracy_score(Ytest, count_svm_predictions, normalize=False)
overall_accuracy = (round(percentage_accuracy*100, 3))
print('Count Overall Accuracy: ' + str(overall_accuracy) + '%')
print('Count Total Correct Classifications: ' + (str(raw_accuracy)) + '/' + (str(len(Ytest))) + '\n')

target_names = ['Business', 'Entertainment', 'Politics', 'Sport', 'Tech']
print('Count Classification Report: \n')
print(classification_report(Ytest, count_svm_predictions, target_names=target_names))

print('Calculating Cross Validation Scores... \n')
count_train_cross_val_classifier = model_selection.cross_val_score(svm_model, xtrain_count, Ytrain, cv=5)
count_test_cross_val_classifier = model_selection.cross_val_score(svm_model, xtest_count, Ytest, cv=5)
print('Cross validation score for training data: ' + str(round(count_train_cross_val_classifier.mean() * 100, 3)) + '%')
print('Cross validation score for testing data: ' + str(round(count_test_cross_val_classifier.mean() * 100, 3)) + '%' + '\n')

print(confusion_matrix(Ytest, count_svm_predictions))
print('\n')

###TFIDF VECTORIZATION AS FEATURE###
tfidf_vectoriser.fit(combined_df['articles'])
xtrain_tfidf = tfidf_vectoriser.transform(Xtrain)
xtest_tfidf = tfidf_vectoriser.transform(Xtest)

print('Tfidf model is being created...\n')
svm_model.fit(xtrain_tfidf, Ytrain)
print('Tfidf model has been created!\n')

print('Predicting using tfidf...\n')
tfidf_svm_predictions = svm_model.predict(xtest_tfidf)

percentage_accuracy = accuracy_score(Ytest, tfidf_svm_predictions)
raw_accuracy = accuracy_score(Ytest, tfidf_svm_predictions, normalize=False)
overall_accuracy = (round(percentage_accuracy*100, 3))
print('Tfidf Overall Accuracy: ' + str(overall_accuracy) + '%')
print('Tfidf Total Correct Classifications: ' + (str(raw_accuracy)) + '/' + (str(len(Ytest))) + '\n')

print('Tfidf Classification Report: \n')
print(classification_report(Ytest, tfidf_svm_predictions, target_names=target_names))

print('Calculating Cross Validation Scores... \n')
tfidf_train_cross_val_classifier = model_selection.cross_val_score(svm_model, xtrain_tfidf, Ytrain, cv=5)
tfidf_test_cross_val_classifier = model_selection.cross_val_score(svm_model, xtest_tfidf, Ytest, cv=5)
print('Cross validation score for training data: ' + str(round(tfidf_train_cross_val_classifier.mean() * 100, 3))+ '%')
print('Cross validation score for testing data: ' + str(round(tfidf_test_cross_val_classifier.mean() * 100, 3)) + '%' + '\n')

print(confusion_matrix(Ytest, tfidf_svm_predictions))
print('\n')

### BIGRAM COUNT VECTORIZATION AS FEATURE###
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words={'english'})
bigram_vectorizer.fit(combined_df['articles'])
xtrain_bigram = bigram_vectorizer.transform(Xtrain)
xtest_bigram = bigram_vectorizer.transform(Xtest)

print('bigram model is being created...\n')
svm_model.fit(xtrain_bigram, Ytrain)
print('bigram model has been created!\n')

print('Predicting using bigram...\n')
bigram_svm_predictions = svm_model.predict(xtest_bigram)

percentage_accuracy = accuracy_score(Ytest, bigram_svm_predictions)
raw_accuracy = accuracy_score(Ytest, bigram_svm_predictions, normalize=False)
overall_accuracy = (round(percentage_accuracy*100, 3))
print('bigram Overall Accuracy: ' + str(overall_accuracy) + '%')
print('bigram Total Correct Classifications: ' + (str(raw_accuracy)) + '/' + (str(len(Ytest))) + '\n')

print('bigram Classification Report: \n')
print(classification_report(Ytest, bigram_svm_predictions, target_names=target_names))

print('Calculating Cross Validation Scores... \n')
bigram_train_cross_val_classifier = model_selection.cross_val_score(svm_model, xtrain_bigram, Ytrain, cv=5)
bigram_test_cross_val_classifier = model_selection.cross_val_score(svm_model, xtest_bigram, Ytest, cv=5)
print('Cross validation score for training data: ' + str(round(bigram_train_cross_val_classifier.mean() * 100, 3))+ '%')
print('Cross validation score for testing data: ' + str(round(bigram_test_cross_val_classifier.mean() * 100, 3)) + '%' + '\n')

print(confusion_matrix(Ytest, bigram_svm_predictions))
print('\n')

print('All done! Please take the time to review the results.')
