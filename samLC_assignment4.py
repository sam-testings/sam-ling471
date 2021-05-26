# Skeleton for Assignment 4.
# Ling471 Spring 2021.

import pandas as pd
import string
import sys

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# These are your own functions you wrote for Assignment 3:
# from evaluation import computePrecisionRecall, computeAccuracy


# Constants
ROUND = 4
GOOD_REVIEW = 1
BAD_REVIEW = 0
ALPHA = 1


# This function will be reporting errors due to variables which were not assigned any value.
# Your task is to get it working! You can comment out things which aren't working at first.
def main(argv):

    # Read in the data. NB: You may get an extra Unnamed column with indices; this is OK.
    # If you like, you can get rid of it by passing a second argument to the read_csv(): index_col=[0].
    data = pd.read_csv(argv[1])
    # print(data.head()) # <- Verify the format. Comment this back out once done.

    # TODO: Change as appropriate, if you stored data differently (e.g. if you put train data first).
    # You may also make use of the "type" column here instead! E.g. you could sort data by "type".
    # At any rate, make sure you are grabbing the right data! Double check with temporary print statements,
    # e.g. print(test_data.head()).

    test_data = data[25000:50000]  # Assuming the first 25,000 rows are test data.

    # Assuming the second 25,000 rows are training data. Double check!
    train_data = data[:25000]

    # TODO: Set the below 4 variables to contain:
    # X_train: the training data; y_train: the training data labels;
    # X_test: the test data; y_test: the test data labels.
    # Access the data frames by the appropriate column names.
    X_train = train_data['review']
    y_train = train_data["label"]
    X_test = test_data['review']
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


if __name__ == "__main__":
    main(sys.argv)
