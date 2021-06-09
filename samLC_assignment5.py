import pandas as pd
import string
import os
import sys

# sklearn is installed via pip: pip install -U scikit-learn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame 

# TODO: Your custom imports here; or copy the functions to here manually.
# from evaluation import computeAccuracy, computePrecisionRecall
# from assignment3_olzama import predictSimplistic

# TODO: You may need to modify assignment 4 if you just had a main() there.
# my_naive_bayes() should take a column as input and return as output 10 floats (numbers)
# representing the metrics.
# from olzama_assignment4 import my_naive_bayes
ROUND = 4
GOOD_REVIEW = 1
BAD_REVIEW = 0
ALPHA = 1

def my_naive_bayes(col_name, data):
    test_data = data[:25000]  # Assuming the first 25,000 rows are test data.

    # Assuming the second 25,000 rows are training data. Double check!
    train_data = data[25000:50000]

    # TODO: Set the below 4 variables to contain:
    # X_train: the training data; y_train: the training data labels;
    # X_test: the test data; y_test: the test data labels.
    # Access the data frames by the appropriate column names.
    X_train = train_data[col_name]
    y_train = train_data["label"]
    X_test = test_data[col_name]
    y_test = test_data["label"]

    # TODO COMMENT: Look up what the astype() method is doing and add a comment, explaining in your own words,
    # what the next two lines are doing.

    # astype() casts the entire dataframe as the same type. 
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # The next three lines are performing feature extraction and word counting. 
    # They are choosing which words to count frequencies for, basically, to discard some of the noise.
    # If you are curious, you could read about TF-IDF,
    # e.g. here: https://www.geeksforgeeks.org/tf-idf-model-for-page-ranking/
    # TODO: Add a general brief comment on why choosing which words to count may be important.

    # Not all words may be relevant to the task at hand and counting all words wouldn't be helpful. 
    # Additionally, certain words may be used in different contexts which is why it may be important to choose which words to count. 
    tf_idf_vect = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_train = tf_idf_vect.fit_transform(X_train.values)
    tf_idf_test = tf_idf_vect.transform(X_test.values)

    # TODO COMMENT: The hyperparameter alpha is used for Laplace Smoothing.
    # Add a brief comment, trying to explain, in your own words, what smoothing is for.

    # Smoothing is to prevent the model from overfitting on the training data. 
    clf = MultinomialNB(alpha=ALPHA)
    # TODO COMMENT: Add a comment explaining in your own words what the "fit()" method is doing.

    # the fit method takes in training data and makes a model based on the data and the targets
    clf.fit(tf_idf_train, y_train)

    # TODO COMMENT: Add a comment explaining in your own words what the "predict()" method is doing in the next two lines.

    # the predict method is testing the model on two seperate sets of data and seeing how well the model does
    y_pred_train = clf.predict(tf_idf_train)
    y_pred_test = clf.predict(tf_idf_test)

    # TODO: Compute accuracy, precision, and recall, for both train and test data.
    # Import and call your methods from evaluation.py which you wrote for HW2.
    # NB: If you methods there accept lists, you may need to cast your pandas label objects to simple python lists:
    # e.g. list(y_train) -- when passing them to your accuracy and precision and recall functions.

    test_correct = 0
    for i in range(len(y_pred_test)):
        if y_pred_test[i] == list(y_test)[i]:
            test_correct += 1
    accuracy_test = test_correct / len(y_pred_test)

    train_correct = 0
    for i in range(len(y_pred_train)):
        if y_pred_train[i] == list(y_train)[i]:
            train_correct += 1
    accuracy_train = train_correct / len(y_pred_train)

    test_tp = 0
    test_pos_total = 0
    for i in range(len(y_pred_test)):
        if y_pred_test[i] == 1:
            if y_pred_test[i] == list(y_test)[i]:
                test_tp += 1
            test_pos_total += 1

    test_tn = 0
    test_neg_total = 0
    for i in range(len(y_pred_test)):
        if y_pred_test[i] == 0:
            if y_pred_test[i] == list(y_test)[i]:
                test_tn += 1
            test_neg_total += 1

    test_fn = test_pos_total - test_tp
    test_fp = test_neg_total - test_tn

    precision_pos_test = test_tp / (test_tp + test_fp)
    recall_pos_test = test_tp / (test_tp + test_fn)

    precision_neg_test = test_tn / (test_tn + test_fn)
    recall_neg_test = test_tn / (test_tn + test_fp)

    train_tp = 0
    train_pos_total = 0
    for i in range(len(y_pred_train)):
        if y_pred_train[i] == 1:
            if y_pred_train[i] == list(y_train)[i]:
                train_tp += 1
            train_pos_total += 1

    train_tn = 0
    train_neg_total = 0
    for i in range(len(y_pred_train)):
        if y_pred_train[i] == 0:
            if y_pred_train[i] == list(y_train)[i]:
                train_tn += 1
            train_neg_total += 1

    train_fn = train_pos_total - train_tp
    train_fp = train_neg_total - train_tn

    precision_pos_train = train_tp / (train_tp + train_fp)
    recall_pos_train = train_tp / (train_tp + train_fn)

    precision_neg_train = train_tn / (train_tn + train_fn)
    recall_neg_train = train_tn / (train_tn + train_fp)

    # Report the metrics via standard output.
    # Please DO NOT modify the format (for grading purposes).
    # You may change the variable names of course, if you used different ones above.
    metrics = []
    metrics.append(accuracy_train)
    metrics.append(precision_pos_train)
    metrics.append(recall_pos_train)
    metrics.append(precision_neg_test)
    metrics.append(recall_neg_train)
    metrics.append(accuracy_test)
    metrics.append(precision_pos_test)
    metrics.append(recall_pos_test)
    metrics.append(precision_neg_test)
    metrics.append(recall_neg_test)
    print(col_name)
    print("Train accuracy:           \t{}".format(round(accuracy_train, ROUND)))
    print("Train precision positive: \t{}".format(
        round(precision_pos_train, ROUND)))
    print("Train recall positive:    \t{}".format(
        round(recall_pos_train, ROUND)))
    print("Train precision negative: \t{}".format(
        round(precision_neg_train, ROUND)))
    print("Train recall negative:    \t{}".format(
        round(recall_neg_train, ROUND)))
    print("Test accuracy:            \t{}".format(round(accuracy_test, ROUND)))
    print("Test precision positive:  \t{}".format(
        round(precision_pos_test, ROUND)))
    print("Test recall positive:     \t{}".format(
        round(recall_pos_test, ROUND)))
    print("Test precision negative:  \t{}".format(
        round(precision_neg_test, ROUND)))
    print("Test recall negative:     \t{}".format(
        round(recall_neg_test, ROUND)))
    return metrics

def main(argv):
    data = pd.read_csv('my_imdb_expanded.csv', index_col=[0])
    # print(data.head())  # <- Verify the format. Comment this back out once done.

    # Part II:
    # Run all models and store the results in variables (dicts).
    # TODO: Make sure you imported your own naive bayes function and it works properly with a named column input!
    # TODO: See also the next todo which gives an example of a convenient output for my_naive_bayes()
    # which you can then easily use to collect different scores.
    # For example (and as illustrated below), the models (nb_original, nb_cleaned, etc.) can be not just lists of scores
    # but dicts where each score will be stored by key, like [TEST][POS][RECALL], etc.
    # But you can also just use lists, except then you must not make a mistake, which score you are accessing,
    # when you plot graphs.
    nb_original = my_naive_bayes('review', data)
    nb_cleaned = my_naive_bayes('cleaned_review', data)
    nb_lowercase = my_naive_bayes('lowercased', data)
    nb_no_stop = my_naive_bayes('no stopwords', data)
    nb_lemmatized = my_naive_bayes('lemmatized', data)

    # Collect accuracies and other scores across models.
    # TODO: Harmonize this with your own naive_bayes() function!
    # The below assumes that naive_bayes() returns a fairly complex dict of scores.
    # (NB: The dicts there contain other dicts!)
    # The return statement for that function looks like this:
    # return({'TRAIN': {'accuracy': accuracy_train, 'POS': {'precision': precision_pos_train, 'recall': recall_pos_train}, 'NEG': {'precision': precision_neg_train, 'recall': recall_neg_train}}, 'TEST': {'accuracy': accuracy_test, 'POS': {'precision': precision_pos_test, 'recall': recall_pos_test}, 'NEG': {'precision': precision_neg_test, 'recall': recall_neg_test}}})
    # This of course assumes that variables like "accuracy_train", etc., were assigned the right values already.
    # You don't have to do it this way; we are giving it to you just as an example.
    train_accuracies = [0.9703, 0.9703, 0.9702, 0.9869, 0.9833]
    train_precision_pos = [0.9587, 0.9594, 0.9594, 0.9846, 0.9803]
    train_recall_pos = [0.9814, 0.9805, 0.9805, 0.9891, 0.9862]
    train_precision_neg = [0.9818, 0.9809, 0.9809, 0.9891, 0.9863]
    train_recall_neg = [0.9595, 0.9603, 0.9603, 0.9847, 0.9804]
    test_accuracies = [0.8683, 0.8691, 0.8691, 0.8641, 0.8582]
    test_precision_pos = [0.8255, 0.8284, 0.8284, 0.8398, 0.8366]
    test_recall_pos = [0.9028, 0.9018, 0.9018, 0.8827, 0.8743]
    test_precision_neg = [0.9111, 0.9098, 0.9098, 0.8884, 0.8798]
    test_recall_neg = [0.8393, 0.8413, 0.8413, 0.8473, 0.8434]

    col_names = ["original", "cleaned", "lowercase", "no_stop", "lemmatized"]
    df = DataFrame(list(zip(col_names,train_accuracies, train_precision_pos,train_recall_pos,train_precision_neg,train_recall_neg,test_accuracies,test_precision_pos,test_recall_pos,test_precision_neg,test_recall_neg)))
    print(df)
    legend = ["train_accuracies", "train_precision_pos","train_recall_pos","train_precision_neg","train_recall_neg","test_accuracies","test_precision_pos","test_recall_pos","test_precision_neg","test_recall_neg"]

    plt.figure()
    df.plot.bar()
    plt.legend(legend, bbox_to_anchor=(1.4,1), loc=1)
    plt.xlabel("Model")
    plt.ylabel("percent")
    plt.title("accuracy, precision, recall of various models")
    loc, lables = plt.xticks()
    plt.xticks(loc, col_names)
    plt.savefig('models.png',bbox_inches="tight")
    # TODO: Initialize other score lists similarly. The precision and recalls, for negative and positive, train and test.
    
    '''
    for model in [nb_original, nb_cleaned, nb_lowercase, nb_no_stop, nb_lemmatized]:
        # TODO: See comment above about where this "model" dict comes from.
        # If you are doing something different, e.g. just a list of scores,
        # that's fine, change the below as appropriate,
        # just make sure you don't confuse where which score is.
        train_accuracies.append(model[0])
        test_accuracies.append(model[5])
        # TODO: Collect other scores similarly. The precision and recalls, for negative and positive, train and test.
        train_precision_pos.append(model[1])
        train_recall_pos.append(model[2])
        train_precision_neg.append(model[3])
        train_recall_neg.append(model[4])
        test_precision_pos.append(model[6])
        test_recall_pos.append(model[7])
        test_precision_neg.append(model[8])
        test_recall_neg.append(model[9])
    '''
    # TODO: Create the plot(s) that you want for the report using matplotlib (plt).
    # Use the below to save pictures as files:
    # plt.savefig('filename.png')


if __name__ == "__main__":
    main(sys.argv)
