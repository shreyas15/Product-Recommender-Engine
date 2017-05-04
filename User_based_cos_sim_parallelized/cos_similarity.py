# author: shreyas s bhat
# email: ssubra15@uncc.edu

import sys
import random
import numpy as np
import pdb
from collections import defaultdict
from itertools import combinations
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg.distributed import MatrixEntry
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import Vectors

# custom helper functions

def nearestNeighbors(user,user_similarities,n):
    "To sort predictions list by similarity and select the top-N neighbors"
    user_similarities.sort(key=lambda x: x[1][0],reverse=True)
    return user, user_similarities[:n]

def topNRecommendations(user_id,user_sims,users_with_rating,n):
    "To calculate recommendations using weighted sums method"
    "Return N recommendations"

    # initialize dictionary to store the score of each individual product,
    # a product can exist in more than one item neighborhood
    total = defaultdict(int)
    similarity_sums = defaultdict(int)

    for (neighbor,(sim,count)) in user_sims:
        unscored_prods = users_with_rating.get(neighbor,None)            # lookup the product predictions for this neighbor
        if unscored_prods:
            for (prod,rating) in unscored_prods:
                if neighbor != prod:
                    total[neighbor] += sim * rating                     # update total and similarity_sums with the rating data
                    similarity_sums[neighbor] += sim

    # normalized list of scored products
    # using .items() we convert totals to a to a list of (prod, total_)
    scored_prods = [(total_/similarity_sums[prod],prod) for prod,total_ in total.items()]
    print scored_prods

    # sort the scored products in ascending order
    scored_prods.sort(reverse=True)

    # remove the product score
    ranked_prods = [x[1] for x in scored_prods]

    return user_id,ranked_prods[:n]

def cos_sim(user_pair,rating_pairs):
    "For each user1-user2 pair, compute the cosine similarity measure"
    "append co_raters_count to the returned value"

    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    for rating_pair in rating_pairs:
        sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
        n += 1
    cos_sim_measure = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))
    return user_pair, (cos_sim_measure,n)

def cosine(dot_product,normalized_sq_rating1,normalized_sq_rating2):
    "The cosine between two vectors X, Y = dotProduct(X, Y) / (norm(X) * norm(Y))"
    num = dot_product
    denom = normalized_sq_rating1 * normalized_sq_rating2
    return (num / (float(denom))) if denom else 0.0

def findUserPairs(product_id, user_rating_tuple):
    "For each product, find all user1-user2 pairs"
    "and corresponding rating1-rating2 pairs"
    for user1, user2 in combinations(user_rating_tuple,2):
        return (user1[0],user2[0]),(user1[1],user2[1])

def vectorizeProducts(line):
    "Parse each line assuming a | delimiter. Key is user_id"
    line = line.split("|")
    return line[0],(line[1],float(line[2]))

def sampleUserRatings(product_id, user_rating_tuple, n):
    "For number of products rated > n"
    "replace their rating history with a sample of n user_rating_tuple"
    if len(user_rating_tuple) > n:
        return product_id, random.sample(user_rating_tuple,n) #use random sampling
    else:
        return product_id, user_rating_tuple

def firstUserKey(user_pair,product_sim_data):
    "For each user1-user2 pair, make the user1 id the key"
    (user1_id,user2_id) = user_pair
    return user1_id,(user2_id,product_sim_data)

def parseProduct(line):
    "Parses records in metadata with format product_id#title"
    fields = line.strip().split("#")
    return int(fields[0]), fields[1]


if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print "Usage: spark-submit --driver-memory 2g recommenderEngine.py <ratings_file> <meta_data_file> <single_user_id>"
        sys.exit(1)

    # set up environment
    conf = SparkConf().setAppName("CosineSimilarity").set("spark.executor.memory", "8g").set("spark.driver.maxResultSize", "8g")
    sc = SparkContext(conf=conf)

    lines = sc.textFile(sys.argv[1])
    meta_lines = sc.textFile(sys.argv[2])
    query_user = sys.argv[3]
    prod_dict = dict(meta_lines.map(parseProduct).collect())

    print ('Environment set and input file read.')
    # we make use of the spark broadcast feature and send the product purchase history
    # for each user across all nodes.
    user_prod_purchases = lines.map(vectorizeProducts).groupByKey().collect()
    # create a user-product dictionary to reference user vs products
    user_prod_dict = {}
    for (user,items) in user_prod_purchases:
        user_prod_dict[user] = items
    user_prod_broadcast = sc.broadcast(user_prod_dict)
    print ('Broadcast data to all nodes in the cluster - Complete.')

    # now we need to obtain a sparse matrix with product vs user_id, rating combination
    # for this we map each line using the vectorizeProducts() function.
    # cache this for faster access later on
    product_user_pairs = lines.map(vectorizeProducts).groupByKey().map(lambda x: sampleUserRatings(x[0],x[1],500)).cache()
    print ('Finished finding user-rating pairs for each product.')

    # for pair-wise users - u1 and u2 we make rating_u1, ratings_u2 tuples`
    pairwise_users = product_user_pairs.filter(lambda x: len(x[1]) > 1).map(lambda x: findUserPairs(x[0],x[1])).groupByKey()
    print ('Finished finding user-user pairs.')

    # now that we have user pairs, we have to calculate the cosine similarity between them
    # we select the top N-neighbors
    user_similarities = pairwise_users.map(
        lambda p: cos_sim(p[0],p[1])).map(
        lambda p: firstUserKey(p[0],p[1])).groupByKey().map(
        lambda x : (x[0], list(x[1]))).map(
        lambda p: nearestNeighbors(p[0],p[1],50))

    print ('Making recommendations... please wait.')
    # make top 100 recommendations to each user
    recommendations = user_similarities.map(lambda x: topNRecommendations(x[0],x[1],user_prod_broadcast.value,100)).collect()

    # print recommendations for the queried user
    recom_found = True
    for x in recommendations:
        if (x[0] == str(query_user)):
            for index, y in enumerate(x[1]):
                print ("#%d ID: %s: %s" % (index, y, prod_dict[int(y[0])]))
            recom_found = False
    if recom_found:
        print "No recommendations found"

    sc.stop()
