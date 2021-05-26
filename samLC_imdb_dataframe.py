# Skeleton for Assignment 4, Part 1.
# Ling471 Spring 2021.

import sys
import re
import string
from pathlib import Path

import pandas as pd


'''
Write a function which accepts a list of 4 directories:
train/pos, train/neg, test/pos, and test/neg.

The result of calling this program on the 4 directories is a new .csv file in the working directory.
'''

# Constants:
POS = 1
NEG = 0


def createDataFrame(argv):
    new_filename = "my_imdb_dataframe.csv"
    # TODO: Create a single dataframe from the 4 IMBD directories (passed as argv[1]--argv[4]).
    # For example, "data" can be a LIST OF LISTS.
    # In this case, each list is a set of column values, e.g. ["0_2.txt", "neg", "test", "Once again Mr Costner..."]
    # You may use a different way of creating a dataframe so long as the result is accurate.
    # TODO: Call the cleanFileContents() function on each file, as you are iterating over them.
    data = []


    # Your code here...
    # Try to create a list of lists, for example, as illustrated above.
    # Consider writing a separate function which takes a filename and returns a list representing the reivew vector.
    # This will make your code here cleaner.

    for i in range(1, 5):
        path = Path(argv[i])
        for filename in path.iterdir():
            if filename.is_file() and filename.suffix == '.txt':
                row = []
                clean_text = cleanFileContents(filename)
                dir_name = argv[i].split('/')
                label = 0
                if dir_name[8] == "pos":
                    label = 1
                type = dir_name[0]
                row.append(filename)
                row.append(label)
                row.append(type)
                row.append(clean_text)
                data.append(row)
    # Once you are done, the below code will only require modifications if your data variable is not a list of lists.
    # Sample column names; you can use different ones if you prefer,
    # but then make sure to make appropriate changes in assignment4_skeleton.py.
    column_names = ["file", "label", "type", "review"]
    # Sample way of creating a dataframe. This assumes that "data" is a LIST OF LISTS.
    df = pd.DataFrame(data=data, columns=column_names)
    print(df)
    # Saving to a file:
    df.to_csv(new_filename)


'''
The below function should be called on a file name.
It opens the file, reads its contents, stores it in a variable.
Then it removes punctuation marks, and returns the "cleaned" text.
'''


def cleanFileContents(f):
    with open(f, 'r') as f:
        text = f.read()
    clean_text = text.translate(str.maketrans('', '', string.punctuation))
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text


def main(argv):
    createDataFrame(argv)


if __name__ == "__main__":
    main(sys.argv)
