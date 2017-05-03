import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

def sliceRating(line):
    "Parses a rating record from ratings csv with format user_id|product_id|rating|timestamp "
    fields = line.strip().split("|")
    return (long(fields[3]) / 100) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def retrieveProduct(line):
    "Parses records in metadata with format product_id#title"
    fields = line.strip().split("#")
    return int(fields[0]), fields[1]

def loadRatings(ratingsFile):
    "Load ratings from file "
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [sliceRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
    else:
        return ratings

def calculateRmse(model, data, n):
    "Compute RMSE (Root Mean Squared Error)"
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print "Usage: spark-submit --driver-memory 2g recommenderEngine_ALS.py <ratings_file> <meta_data_file> <single_user_ratings_file>"
        sys.exit(1)

    # set up environment
    conf = SparkConf().setAppName("AmazonProductRecommender").set("spark.executor.memory", "8g").set("spark.driver.maxResultSize", "8g")
    sc = SparkContext(conf=conf)

    # load personal ratings
    myRatings = loadRatings(sys.argv[3])
    myRatingsRDD = sc.parallelize(myRatings, 1)

    # load ratings and product titles
    # ratings is an RDD of (last digit of timestamp, (user_id, product_id, rating))
    ratings = sc.textFile(sys.argv[1]).map(sliceRating)

    # products is an RDD of (product_id, Title)
    products = dict(sc.textFile(sys.argv[2]).map(retrieveProduct).collect())

    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numProducts = ratings.values().map(lambda r: r[1]).distinct().count()

    print "Got %d ratings from %d users on %d products." % (numRatings, numUsers, numProducts)

    # split ratings into train:validate:test = 6:2:2
    # training, validation, test are all RDDs of (user_id, product_id, rating)

    numPartitions = 4
    # training, validation, test = ratings.randomSplit([5,3,2], 0)
    training = ratings.filter(lambda x: x[0] < 6).values().union(myRatingsRDD).repartition(numPartitions).cache()

    validation = ratings.filter(lambda x: x[0] >= 6 and x[0] < 8).values().repartition(numPartitions).cache()

    test = ratings.filter(lambda x: x[0] >= 8).values().cache()

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    print "Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)

    # train models and evaluate them on the validation set
    ranks = [8, 12]
    lambdas = [0.01, 0.1, 10]
    numIters = [10, 20]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1

    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        model = ALS.train(training, rank, numIter, lmbda)
        validationRmse = calculateRmse(model, validation, numValidation)
        print "RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter)
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter

    testRmse = calculateRmse(bestModel, test, numTest)

    # use the test data to evaluate the best model.
    print "The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda) \
      + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse)

    # baseline performance comparison
    meanRating = training.union(validation).map(lambda x: x[2]).mean()
    baselineRmse = sqrt(test.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / numTest)
    improvement = (baselineRmse - testRmse) / baselineRmse * 100
    print "The best model improves the baseline by %.2f" % (improvement) + "%."

    # make personalized recommendations for the given user in the input single user ratings file
    # we eliminate all products user has already rated to make fresh predictions
    myRatedProductIds = set([x[1] for x in myRatings])
    candidates = sc.parallelize([m for m in products if m not in myRatedProductIds])
    predictions = bestModel.predictAll(candidates.map(lambda x: (0, x))).collect()
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:50]

    print "Products recommended for you:"
    for i in xrange(len(recommendations)):
        print ("%2d: %s" % (i + 1, products[recommendations[i][1]])).encode('ascii', 'ignore')

    # clean up
    sc.stop()
