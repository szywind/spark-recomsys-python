import os, sys
import pandas as pd
from pyspark import mllib
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

os.environ['PYSPARK_PYTHON'] = sys.executable



## Step 0: Set up the cluster configurations
conf = (SparkConf().set("spark.master", "spark://YZM-2.local:7077")
        .set("spark.eventLog.enabled)",True)
        .set("spark.task.cpus",1)
        .set("spark.driver.memory","1g")
        .set("spark.executor.cores","4")
        .set("spark.executor.memory","21000m")
        .set("spark.eventLog.dir","/Users/a/PycharmProjects/spark-recomsys")
        .setAppName("Clothes Recommend System"))
sc = SparkContext(conf=conf)

# file = 'hdfs://localhost:9000/clothes/input/clothes-features.csv'
file = 'hdfs://yzm2:9000/demo/input/clothes-features.csv'


HDFS_ROOT = 'hdfs://localhost:9000/clothes/'

HDFS_ROOT = 'hdfs://yzm2:9000/demo/'



class ClothesRecommend:
    def __init__(self):
        pass

    def ModelCF(self):

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

    def ItemCF(self):
        pass
    def UserCF(self):
        pass
    def UserSimilarity(self):
        pass
    def ItemSimiarity(self):
        pass














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
