import os, sys
import pandas as pd
from pyspark import mllib
from pyspark import SparkContext, SparkConf
import numpy as np
from numpy.linalg import norm

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

os.environ['PYSPARK_PYTHON'] = sys.executable

local_test = False

if local_test:
    HDFS_ROOT = "hdfs://localhost:9000/demo/" # set by conf of hadoop
    SPARK_MASTER_ADDR = "local[2]"
    # SPARK_MASTER_ADDR = "spark://YZM-2.local:7077" # read spark ui:8080 or check SPARK_MASTER_HOST in the file $SPARK_HOME/sbin/start-master.sh
    EVENT_LOG_DIR = "/Users/a/PycharmProjects/spark-recomsys/logs"
else:
    HDFS_ROOT = "hdfs://yzm2:9000/demo/"
    SPARK_MASTER_ADDR = "spark://localhost:7077"
    EVENT_LOG_DIR = "/home/yzm/logs"


## Step 0: Set up the cluster configurations
conf = (SparkConf().set("spark.master", SPARK_MASTER_ADDR)
        .set("spark.eventLog.enabled)",True)
        .set("spark.task.cpus",1)
        .set("spark.driver.memory","1g")
        .set("spark.executor.cores","4")
        .set("spark.executor.memory","21000m")
        .set("spark.eventLog.dir", EVENT_LOG_DIR)
        .setAppName("Clothes Recommend System"))
sc = SparkContext(conf=conf)

# class Item:
#     def __init__(self, cid, attr = None, pref = None):
#         self.cid = int(cid[cid.find('_')+1:cid.rfind('.')])
#         self.attr = attr
#         self.pref = pref
#
#     def attribute_similarity(self, that):
#         return np.dot(self.attr, that.attr)/(norm(self.attr)*norm(that.attr))
#
#     def preference_similarity(self, that):
#         return np.dot(self.pref, that.pref)/(norm(self.pref)*norm(that.pref))
#
#
# class User:
#     def __init__(self, uid, attr = None, pref = None):
#         self.uid = int(uid[:uid.rfind('.')])
#         self.attr = attr
#         self.pref = pref
#
#     def attribute_similarity(self, that):
#         return np.dot(self.attr, that.attr)/(norm(self.attr)*norm(that.attr))
#
#     def preference_similarity(self, that):
#         return np.dot(self.pref, that.pref)/(norm(self.pref)*norm(that.pref))


def extract_user_id(uid):
    return int(uid[:uid.rfind('.')])

def extract_item_id(cid):
    return int(cid[cid.find('_') + 1:cid.rfind('.')])

def cosin_similarity(u, v):
    return (np.inner(u, v)/(norm(u)*norm(v)) + 1) / 2.0

class ClothesRecommend:
    def __init__(self):
        rating_data = sc.textFile(HDFS_ROOT + 'input/ratings')
        item_data = sc.textFile(HDFS_ROOT + 'input/clothes-features.csv')
        user_data = sc.textFile(HDFS_ROOT + 'input/user-features.csv')

        self.ratings = rating_data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
        self.items = item_data.map(lambda l: l.split(',')).map(lambda l: (extract_item_id(l[0]), list(map(float, l[1:]))))
        self.users = user_data.map(lambda l: l.split(',')).map(lambda l: (extract_user_id(l[-1]), list(map(float, l[:-1]))))

        self.num_users = self.users.count()
        self.num_items = self.items.count()

        # get {user: {item: rating}}
        self.user_rating_dict = dict(self.ratings.map(lambda l: (int(l[0]), (int(l[1]), float(l[2])))).groupByKey().mapValues(dict).collect())
        # get {item: {user: rating}}
        self.item_rating_dict = dict(self.ratings.map(lambda l: (int(l[1]), (int(l[0]), float(l[2])))).groupByKey().mapValues(dict).collect())

        # https://stackoverflow.com/questions/40087483/spark-average-of-values-instead-of-sum-in-reducebykey-using-scala
        # compute average rating of each item
        self.item_avg_rating = self.ratings.map(lambda l: (int(l[1]), (float(l[2]), 1))).\
            reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1])).mapValues(lambda x: 1.0 * x[0] / x[1]).collectAsMap()

        self.model = None

    def trainModelCF(self):

        # Load and parse the data
        # data = sc.textFile(HDFS_ROOT + 'input/ratings')
        # ratings = self.rating_data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

        # Build the recommendation model using Alternating Least Squares
        rank = 10
        numIterations = 10
        self.model = ALS.train(self.ratings, rank, numIterations)

        # Evaluate the model on training data
        testdata = self.ratings.map(lambda p: (p[0], p[1]))
        predictions = self.model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
        ratesAndPreds = self.ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
        MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
        print("Mean Squared Error = " + str(MSE))

        # Save and load model
        model_path = HDFS_ROOT + 'models/modelCF'
        self.model.save(sc, model_path)
        sameModel = MatrixFactorizationModel.load(sc, model_path)

    def predModelCF(self, uid, cid):
        if self.model is None:
            try:
                model_path = HDFS_ROOT + 'models/modelCF'
                self.model = MatrixFactorizationModel.load(sc, model_path)
            except:
                raise RuntimeError("No model found!")
        rating = self.model.predict(uid, cid)
        return rating


    def predUserCF(self, uid, cid):
        rating = 0
        denominator = 0.0
        if uid in self.user_rating_dict:
            if cid in self.user_rating_dict[uid]:
                return self.user_rating_dict[uid][cid]
        for u in self.user_rating_dict:
            if u == uid:
                continue
            try:
                rating += self.user_rating_dict[u][cid] * self.user_similarity[(uid, u)]
                denominator += self.user_similarity[(uid, u)]
            except:
                pass
        if denominator == 0:
            try:
                rating = self.item_avg_rating[cid]
            except:
                pass
        else:
            rating /= denominator
        return rating

    def predItemCF(self, uid, cid):
        rating = 0
        denominator = 0.0
        if uid in self.user_rating_dict:
            if cid in self.user_rating_dict[uid]:
                return self.user_rating_dict[uid][cid]
            for c in self.user_rating_dict[uid]:
                try:
                    rating += self.user_rating_dict[uid][c] * self.item_similarity[(cid, c)]
                    denominator += self.item_similarity[(cid, c)]
                except:
                    pass
        if denominator == 0:
            try:
                rating = self.item_avg_rating[cid]
            except:
                pass
        else:
            rating /= denominator
        return rating


    def compUserSimilarity(self):

        # compute similarity b/w users
        user_list = self.users.collect()
        self.user_similarity = {}
        for i in range(self.num_users):
            for j in range(i, self.num_users):
                u = user_list[i]
                v = user_list[j]
                self.user_similarity[(u[0], v[0])] = self.user_similarity[(v[0], u[0])] = cosin_similarity(u[1], v[1])

    def compItemSimiarity(self):

        # compute similarity b/w items
        item_list = self.items.collect()
        self.item_similarity = {}
        for i in range(self.num_items):
            for j in range(i, self.num_items):
                u = item_list[i]
                v = item_list[j]
                self.item_similarity[(u[0], v[0])] = self.item_similarity[(v[0], u[0])] = cosin_similarity(u[1], v[1])



if __name__ == "__main__":
    rcm = ClothesRecommend()
    rcm.compItemSimiarity()
    rcm.compUserSimilarity()
    rcm.trainModelCF()

    rcm.predUserCF(12, 3)
    rcm.predItemCF(12, 16)
    rcm.predModelCF(12,16)










'''
import os, sys
from pyspark import mllib
from pyspark import SparkContext, SparkConf
import numpy as np
from numpy.linalg import norm

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
conf = (SparkConf().set("spark.master", "spark://localhost:7077")  
        .set("spark.eventLog.enabled)",True)
        .set("spark.task.cpus",1)
        .set("spark.driver.memory","1g")
        .set("spark.executor.cores","4")
        .set("spark.executor.memory","21000m")
        .set("spark.eventLog.dir","/home/yzm/logs")
        .setAppName("Clothes Recommend System"))
sc = SparkContext(conf=conf)

HDFS_ROOT = 'hdfs://yzm2:9000/demo/'
model_path = HDFS_ROOT + 'models/modelCF'

# Load and parse the data
data = sc.textFile(HDFS_ROOT + 'input/ratings')
ratings = data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
print("Mean Squared Error = " + str(MSE))

# Save and load model
model_path = HDFS_ROOT + 'models/modelCF'

# model.save(sc, model_path)
# sameModel = MatrixFactorizationModel.load(sc, model_path)






def extract_user_id(uid):
    return int(uid[:uid.rfind('.')])

def extract_item_id(cid):
    return int(cid[cid.find('_') + 1:cid.rfind('.')])

def cosin_similarity(u, v):
    return (np.inner(u, v)/(norm(u)*norm(v)) + 1) / 2.0

rating_data = sc.textFile(HDFS_ROOT + 'input/ratings')
item_data = sc.textFile(HDFS_ROOT + 'input/clothes-features.csv')
user_data = sc.textFile(HDFS_ROOT + 'input/user-features.csv')

ratings = rating_data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
items = item_data.map(lambda l: l.split(',')).map(lambda l: (extract_item_id(l[0]), list(map(float, l[1:]))))
users = user_data.map(lambda l: l.split(',')).map(lambda l: (extract_user_id(l[-1]), list(map(float, l[:-1]))))

num_users = users.count()
num_items = items.count()

# get {user: {item: rating}}
user_rating_dict = dict(ratings.map(lambda l: (int(l[0]), (int(l[1]), float(l[2])))).groupByKey().mapValues(dict).collect())
# get {item: {user: rating}}
item_rating_dict = dict(ratings.map(lambda l: (int(l[1]), (int(l[0]), float(l[2])))).groupByKey().mapValues(dict).collect())

# https://stackoverflow.com/questions/40087483/spark-average-of-values-instead-of-sum-in-reducebykey-using-scala
# compute average rating of each item
item_avg_rating = ratings.map(lambda l: (int(l[1]), (float(l[2]), 1))).\
            reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1])).mapValues(lambda x: 1.0 * x[0] / x[1]).collectAsMap()



def predModelCF(uid, cid):
    if model is None:
        try:
            model_path = HDFS_ROOT + 'models/modelCF'
            model = MatrixFactorizationModel.load(sc, model_path)
        except:
            raise RuntimeError("No model found!")
    rating = model.predict(uid, cid)
    return rating


def predUserCF(uid, cid):
    rating = 0
    denominator = 0.0
    if uid in user_rating_dict:
        if cid in user_rating_dict[uid]:
            return user_rating_dict[uid][cid]
    for u in user_rating_dict:
        if u == uid:
            continue
        try:
            rating += user_rating_dict[u][cid] * user_similarity[(uid, u)]
            denominator += user_similarity[(uid, u)]
        except:
            pass
    if denominator == 0:
        try:
            rating = item_avg_rating[cid]
        except:
            pass
    else:
        rating /= denominator
    return rating

def predItemCF(uid, cid):
    rating = 0
    denominator = 0.0
    if uid in user_rating_dict:
        if cid in user_rating_dict[uid]:
            return user_rating_dict[uid][cid]
        for c in user_rating_dict[uid]:
            try:
                rating += user_rating_dict[uid][c] * item_similarity[(cid, c)]
                denominator += item_similarity[(cid, c)]
            except:
                pass
    if denominator == 0:
        try:
            rating = item_avg_rating[cid]
        except:
            pass
    else:
        rating /= denominator
    return rating



# compute similarity b/w users
user_list = users.collect()
user_similarity = {}
for i in range(num_users):
    for j in range(i, num_users):
        u = user_list[i]
        v = user_list[j]
        user_similarity[(u[0], v[0])] = user_similarity[(v[0], u[0])] = cosin_similarity(u[1], v[1])

# compute similarity b/w items
item_list = items.collect()
item_similarity = {}
for i in range(num_items):
    for j in range(i, num_items):
        u = item_list[i]
        v = item_list[j]
        item_similarity[(u[0], v[0])] = item_similarity[(v[0], u[0])] = cosin_similarity(u[1], v[1])


'''