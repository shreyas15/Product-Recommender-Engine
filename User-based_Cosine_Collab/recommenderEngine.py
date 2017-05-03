import pandas as pd
import numpy as np
import scipy as sp
from scipy.sparse import issparse
import operator
import sys
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import fast_dot
from sklearn.metrics.pairwise import check_pairwise_arrays

from pyspark import SparkConf, SparkContext

names = [
    'user_id',
    'product_id',
    'rating',
    'timestamp',
]

def safe_sparse_dot(a, b):
    "Compute dot product between a and b"
    if issparse(a) or issparse(b):
        ret = a * b
        if hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return fast_dot(a, b)

def cosine_similarity(X, Y=None):
    "Compute cosine similarity between samples in X and Y"

    # to avoid recursive import
    X, Y = check_pairwise_arrays(X, Y)
    X_normalized = normalize(X, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, copy=True)
    K = safe_sparse_dot(X_normalized, Y_normalized.T)
    return K

if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print "Usage: spark-submit --driver-memory 2g recommenderEngine.py <ratings_file> <user_id> <product_id>"
        sys.exit(1)

    # set up environment
    conf = SparkConf().setAppName("AmazonProductRecommender").set("spark.executor.memory", "8g").set("spark.driver.maxResultSize", "8g")
    sc = SparkContext(conf=conf)

    input_user_id = int(sys.argv[2])
    input_product_id = str(sys.argv[3])

    rating = pd.read_csv(sys.argv[1])

    merged=rating[['user_id', 'product_id', 'rating']]
    merged_sub= merged[merged.user_id <= 10000]
    #print merged_sub.head()

    piv = merged_sub.pivot_table(index=['user_id'], columns=['product_id'], values='rating')

    # We are subtracting the mean from each rating to standardize and all users with only one rating or who had rated everything the same will be dropped
    # Normalize the values
    piv_norm = piv.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
    #print piv_norm

    # Remove all columns containing only zeros for users who did not rate
    piv_norm.fillna(0, inplace=True)
    piv_norm = piv_norm.T
    piv_norm = piv_norm.loc[:, (piv_norm != 0).any(axis=0)]
    #print piv_norm

    # making a sparse matrix
    piv_sparse = sp.sparse.csr_matrix(piv_norm.values)

    #computing cosine similarities
    item_similarity = cosine_similarity(piv_sparse)
    user_similarity = cosine_similarity(piv_sparse.T)

    # Inserting the similarity matricies into dataframes
    item_sim_df = pd.DataFrame(item_similarity, index = piv_norm.index, columns = piv_norm.index)
    user_sim_df = pd.DataFrame(user_similarity, index = piv_norm.columns, columns = piv_norm.columns)
    #print user_sim_df

    # This function will return the top 10 products with the highest cosine similarity value
    def top_prods(product_id):
        count = 1
        print('Similar products to {} include:'.format(product_id))
        for item in item_sim_df.sort_values(by = product_id, ascending = False).index[1:11]:
            print('No. {}: {}'.format(count, item))
            count +=1
        print ('\n')

    # This function will return the top 10 users with the highest similarity value
    def top_users(user):

        if user not in piv_norm.columns:
            return('No data available on user {}'.format(user))

        print('Most Similar Users:')
        sim_values = user_sim_df.sort_values(by=user, ascending=False).loc[:,user].tolist()[1:11]
        sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:11]
        zipped = zip(sim_users, sim_values,)
        for user, sim in zipped:
            print('User #{0}, Similarity value: {1:.2f}'.format(user, sim))
        print ('\n')

    # This function constructs a list of the highest rated products for the given user based on similar users
    # returns product_id along with the frequency of appearance
    def similar_user_recs(user_id):
        if user_id not in piv_norm.columns:
            return('No data available on user {}'.format(user_id))

        sim_users = user_sim_df.sort_values(by=user_id, ascending=False).index[1:11]
        best = []
        most_common = {}

        for i in sim_users:
            max_score = piv_norm.loc[:, i].max()
            best.append(piv_norm[piv_norm.loc[:, i]==max_score].index.tolist())
        for i in range(len(best)):
            for j in best[i]:
                if j in most_common:
                    most_common[j] += 1
                else:
                    most_common[j] = 1
        sorted_list = sorted(most_common.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_list[:5]

    # This function calculates the weighted average of similar users to predicted rating for an input user and product
    def predicted_rating(product_id, user_id):
        sim_users = user_sim_df.sort_values(by=user_id, ascending=False).index[1:1000]
        user_values = user_sim_df.sort_values(by=user_id, ascending=False).loc[:,user_id].tolist()[1:1000]
        rating_list = []
        weight_list = []
        for j, i in enumerate(sim_users):
            rating = piv.loc[i, product_id]
            similarity = user_values[j]
            if np.isnan(rating):
                continue
            elif not np.isnan(rating):
                rating_list.append(rating*similarity)
                weight_list.append(similarity)
        return sum(rating_list)/sum(weight_list)

    top_prods(input_product_id)

    top_users(input_user_id)

    print ('List of products based on other user ratings for the current user: ')
    print similar_user_recs(input_user_id)
    print ('\n')

    watched = piv.T[piv.loc[input_user_id,:]>0].index.tolist()
    errors = []
    for i in watched:
        actual=piv.loc[input_user_id, i]
        predicted = predicted_rating(i, input_user_id)
        errors.append((actual-predicted)**2)

    print "RMSE is: ", np.mean(errors)

    sc.stop()
