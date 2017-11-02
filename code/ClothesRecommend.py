import os, sys
import pandas as pd
from pyspark import mllib
from pyspark import SparkContext, SparkConf
import numpy as np
from numpy.linalg import norm

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating,

os.environ['PYSPARK_PYTHON'] = sys.executable

local_test = False

if local_test:
    HDFS_ROOT = "hdfs://localhost:9000/clothes/" # set by conf of hadoop
    SPARK_MASTER_ADDR = "spark://YZM-2.local:7077"
    EVENT_LOG_DIR = "/Users/a/PycharmProjects/spark-recomsys/logs"
else:
    HDFS_ROOT = "hdfs://yzm2:9000/demo/"
    SPARK_MASTER = "spark://localhost:7077"
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

class Item:
    def __init__(self, cid, attr = None, prefer = None):
        self.uid = int(cid[cid.find('_')+1:cid.rfind('.')])
        self.attr = attr
        self.prefer = prefer

    def attribute_similarity(self, that):
        return np.dot(self.attr, that.feature)/(norm(self.attr)*norm(that.feature))

    def preference_similarity(self, that):
        return np.dot(self.prefer, that.prefer)/(norm(self.prefer)*norm(that.prefer))


class User:
    def __init__(self, uid, attr = None, prefer = None):
        self.uid = int(uid[:uid.rfind('.')])
        self.attr = attr
        self.prefer = prefer

    def attribute_similarity(self, that):
        return np.dot(self.attr, that.feature)/(norm(self.attr)*norm(that.feature))

    def preference_similarity(self, that):
        return np.dot(self.prefer, that.prefer)/(norm(self.prefer)*norm(that.prefer))



class ClothesRecommend:
    def __init__(self):
        rating_data = sc.textFile(HDFS_ROOT + 'input/ratings')
        item_data = sc.textFile(HDFS_ROOT + 'input/clothes-features.csv')
        user_data = sc.textFile(HDFS_ROOT + 'input/user-features.csv')

        self.ratings = rating_data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
        self.items = item_data.map(lambda l: l.split(',')).map(lambda l: Item(l[0], attr=l[1:]))
        self.users = user_data.map(lambda l: l.split(',')).map(lambda l: User(l[-1], attr=l[:-1]))

        rating_dict = {}
        for r in self.ratings.collect():
            if not r.user in rating_dict:
                rating_dict[r.user] = {}
            rating_dict[r.user][r.product] = r.rating

    def trainModelCF(self):

        # Load and parse the data
        # data = sc.textFile(HDFS_ROOT + 'input/ratings')
        # ratings = self.rating_data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

        # Build the recommendation model using Alternating Least Squares
        rank = 10
        numIterations = 10
        model = ALS.train(self.ratings, rank, numIterations)

        # Evaluate the model on training data
        testdata = self.ratings.map(lambda p: (p[0], p[1]))
        predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
        ratesAndPreds = self.ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
        MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
        print("Mean Squared Error = " + str(MSE))

        # Save and load model
        model_path = HDFS_ROOT + 'models/modelCF'
        model.save(sc, model_path)
        sameModel = MatrixFactorizationModel.load(sc, model_path)

    def trainItemCF(self):

        user_list = self.users.collect()
        rating_dict = A
        for user in user_list:
            user.prefer = get_preference(user, A)

        pass

    def trainUserCF(self):
        pass


    def trainUserSimilarity(self):

        # compute similarity b/w users
        user_list = self.users.collect()
        num_user = self.users.count()
        self.user_similarity = {}
        for i in range(num_user):
            for j in range(i, num_user):
                u = user_list[i]
                v = user_list[j]
                self.user_similarity[(u.uid, v.uid)] = self.user_similarity[(v.uid, u.uid)] = u.attribute_similarity(v)

    def trainItemSimiarity(self):

        # compute similarity b/w items
        item_list = self.items.collect()
        num_item = self.items.count()
        self.item_similarity = {}
        for i in range(num_item):
            for j in range(i, num_item):
                u = item_list[i]
                v = item_list[j]
                self.item_similarity[(u.uid, v.uid)] = self.item_similarity[(v.uid, u.uid)] = u.attribute_similarity(v)













import os, sys, random
import pandas as pd
from pyspark import mllib
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


conf = (SparkConf().set("spark.master", "spark://localhost:7077")  # read spark ui:8080 or check SPARK_MASTER_HOST in the file $SPARK_HOME/sbin/start-master.sh
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

model.save(sc, model_path)
sameModel = MatrixFactorizationModel.load(sc, model_path)
