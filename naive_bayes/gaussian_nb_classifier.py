from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def classify(features_train, labels_train):
    ### create classifier
    clf = GaussianNB()
    
    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)
    
    ### return the fit classifier
    return clf

def NBAccuracy(features_train, labels_train, features_test, labels_test):
    ### create and train classifier
    clf = classify(features_train, labels_train)
    
    ### predict labels for the test set
    predictions = clf.predict(features_test)
    
    ### calculate and return accuracy
    accuracy = accuracy_score(labels_test, predictions)
    return accuracy
print(f"Model accuracy: {accuracy}")
