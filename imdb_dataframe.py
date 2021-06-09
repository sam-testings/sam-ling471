# Olga Zamaraeva's solution for Assignment 3.
# Ling471 Spring 2021.

import sys
import re
import string
from pathlib import Path

import pandas as pd
import csv

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk import stem
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Constants:
POS = 1
NEG = 0

'''
args
"/Users/samuel/Documents/uw/LING 471/aclImdb/test/neg",
"/Users/samuel/Documents/uw/LING 471/aclImdb/test/pos",
"/Users/samuel/Documents/uw/LING 471/aclImdb/train/neg",
"/Users/samuel/Documents/uw/LING 471/aclImdb/train/pos"
'''

def review_to_words(review, remove_stopwords=False, lemmatize=False):
    # Getting an off-the-shelf list of English "stopwords"
    stops = stopwords.words('english')
    # Initializing an instance of the NLTK stemmer/lemmatizer class
    sno = stem.SnowballStemmer('english')
    # Removing HTML using BeautifulSoup preprocessing package
    review_text = BeautifulSoup(review).get_text()
    # Remove non-letters using a regular expression
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # Tokenizing by whitespace
    words = review_text.split()
    # Recall "list comprehension" from the lecture demo and try to understand what the below loops are doing:
    if remove_stopwords:
        words = [w for w in words if not w in stops]
    if lemmatize:
        lemmas = [sno.stem(w).encode('utf8') for w in words]
        # The join() function is a built-in method of strings.
        # The below says: iterate over the "lemmas" list and create
        # a new string where each item in "lemmas" is added to this new string,
        # and the items are separated by a space.
        # The b-thing is a quirk of the SnowballStemmer package.
        return b" ".join(lemmas)
    else:
        return ' '.join(words)


def cleanFileContents(f):
    with open(f, 'r') as f:
        text = f.read()
    cleaned_text = review_to_words(text)
    lowercased = cleaned_text.lower()
    no_stop = review_to_words(lowercased, remove_stopwords=True)
    lemmatized = review_to_words(no_stop, lemmatize=True)
    return (text, cleaned_text, lowercased, no_stop, lemmatized)


def processFileForDF(f, table, label, t):
    text, cleaned_text, lowercased, no_stop, lemmatized = cleanFileContents(f)
    table.append([f.stem+'.txt', label, t, text,
                 cleaned_text, lowercased, no_stop, lemmatized])


def createDataFrames(argv):
    train_pos = list(Path(argv[1]).glob("*.txt"))
    train_neg = list(Path(argv[2]).glob("*.txt"))
    test_pos = list(Path(argv[3]).glob("*.txt"))
    test_neg = list(Path(argv[4]).glob("*.txt"))

    data = []

    # TODO: Your function from assignment 4, adapted for assignment 5 as needed, goes here.
    # Do all the required preprocerssing.
    #
    for i in range(1, 5):
        path = Path(argv[i])
        counter = 0
        for filename in path.iterdir():
            if filename.is_file() and filename.suffix == '.txt':
                row = []
                text, cleaned_text, lowercased, no_stopwords, lemmatized = cleanFileContents(filename)
                dir_name = argv[i].split('/')
                label = 0
                if dir_name[8] == "pos":
                    label = 1
                type = dir_name[0]
                row.append(filename)
                row.append(label)
                row.append(type)
                row.append(text)
                row.append(cleaned_text)
                row.append(lowercased)
                row.append(no_stopwords)
                row.append(lemmatized)
                data.append(row)
            if counter % 100 == 0:
                print("Processing directory " + str(i) + " out of 4; file " + str(counter) + " out of 12500")
            counter += 1
    # TODO: The program will now be noticeably slower!
    # To reassure yourself that the program is doing something, insert print statements as progress indicators.
    # For example, for each 100th file, print out something like:
    # "Processing directory 1 out of 4; file 99 out of 12500".
    # The enumerate method iterates over both the items in the list and their indices, at the same time.
    # Step through in the debugger to see what i and f are at step 1, step 2, and so forth.

    # Your code goes here... Example of how to get not only a list element but also its index, below:
    # for index, element in enumerate(['a','b','c','d']):
    #    print("{}'s index is {}".format(element,index))

    # Use the below column names if you like:
    column_names = ["file", "label", "type", "review",
                    "cleaned_review", "lowercased", "no stopwords", "lemmatized"]
    df = pd.DataFrame(data=data, columns=column_names)
    df.sort_values(by=['type', 'file'])
    df.to_csv('my_imdb_expanded.csv')


def main(argv):
    createDataFrames(argv)


if __name__ == "__main__":
    main(sys.argv)
