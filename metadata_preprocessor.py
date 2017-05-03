# metadata_preprocessor.py
#
# Standalone Python/Spark program to perform data pre-processing..
# Reads Ratings data and meta data to combine where necessary
# and encode labels to a form fit for processing.
#
#
# Usage: spark-submit data_preprocessor.py <inputdatafile>
# Example usage: spark-submit data_preprocessor.py ratings.csv
#
#

import sys
import pandas as pd
import numpy as np
import csv
import gzip

from sklearn import preprocessing
from pyspark import SparkContext, SparkConf, SQLContext


conf = (SparkConf().set("spark.driver.maxResultSize", "8g"))

#to read data from gzip files
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
      yield eval(l)

#make a dataframe
def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
      df[i] = d
      i += 1
    return pd.DataFrame.from_dict(df, orient='index')

names = [
    'user_id',
    'product_id',
    'rating',
    'timestamp',
]

def labelEncoder(in_csv):
    "This function converts categorical data to numerical values in the supplied dataframe"
    #using pandas read the csv and append column names from names
    # input_data = pd.read_csv(in_csv, sep=",", names=names)
    input_data = pd.read_csv(in_csv, sep=",")
    #print input_data.head()
    #
    user_id_en = preprocessing.LabelEncoder()
    product_id_en = preprocessing.LabelEncoder()

    user_id_en.fit(input_data.user_id)
    product_id_en.fit(input_data.product_id)
    encoded_df = input_data

    encoded_df.user_id = user_id_en.transform(input_data.user_id)
    encoded_df.product_id = product_id_en.transform(input_data.product_id)
    #encoded_df.to_csv('encoded_data_w_index_headers.csv', sep='::',index = False)
    encoded_df.to_csv('ratings_als.csv', sep='|', index = False, header=None)

    #return encoded_df
    #return input_data

if __name__ == "__main__":
    # if len(sys.argv) !=3:
    #     print >> sys.stderr, "Usage: data_preprocessor <ratings_file> <metadata_gzip_file>"
    #     exit(-1)

    sc = SparkContext(appName="DataProcessor", conf=conf)
    sqlContext = SQLContext(sc)


    ## Use this if the file being read is a JSON that is gzipped.
    metadata_df = getDF(sys.argv[1])
    metadata_df.rename(columns={'asin': 'product_id'}, inplace=True)
    metadata_df.drop('description', axis=1, inplace=True)
    metadata_df.drop('price', axis=1, inplace=True)
    metadata_df.drop('salesRank', axis=1, inplace=True)
    metadata_df.drop('imUrl', axis=1, inplace=True)
    metadata_df.drop('brand', axis=1, inplace=True)
    metadata_df.drop('related', axis=1, inplace=True)
    #metadata_df.to_csv('metadata.csv', sep=',')
    metadata_df.to_csv('temp_metadata.csv', sep=',', index = False)

    #labelEncoder(sys.argv[1])
    #labelEncoder(temp_metadata.csv)
    # input_df.drop('timestamp', axis=1, inplace=True)
    # input_df.to_csv('input.csv', sep=',', index = False)


    sc.stop()
