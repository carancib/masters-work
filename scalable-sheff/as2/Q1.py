import pandas as pd
import matplotlib.pyplot as plt
import pyspark  
from pyspark.sql.types import StructField, DoubleType, StructType
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Binarizer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import DecisionTreeClassifier , LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.regression import DecisionTreeRegressor
import time
import numpy as np

# create session

spark = SparkSession.builder \
    .appName("COM6012 Assignment 2") \
    .config("spark.local.dir","/fastdata/acp18ca") \
    .getOrCreate()


sc = spark.sparkContext
sc.setLogLevel("WARN")
    
begin = time.time()

# read data

schema = StructType([StructField('label', DoubleType(), True),
                     StructField('low_1', DoubleType(), True),
                     StructField('low_2', DoubleType(), True),
                     StructField('low_3', DoubleType(), True),
                     StructField('low_4', DoubleType(), True),
                     StructField('low_5', DoubleType(), True),
                     StructField('low_6', DoubleType(), True),
                     StructField('low_7', DoubleType(), True),
                     StructField('low_8', DoubleType(), True),
                     StructField('low_9', DoubleType(), True),
                     StructField('low_10', DoubleType(), True),
                     StructField('low_11', DoubleType(), True),
                     StructField('low_12', DoubleType(), True),
                     StructField('low_13', DoubleType(), True),
                     StructField('low_14', DoubleType(), True),
                     StructField('low_15', DoubleType(), True),
                     StructField('low_16', DoubleType(), True),
                     StructField('low_17', DoubleType(), True),
                     StructField('low_18', DoubleType(), True),
                     StructField('low_19', DoubleType(), True),
                     StructField('low_20', DoubleType(), True),
                     StructField('low_21', DoubleType(), True),
                     StructField('high_1', DoubleType(), True),
                     StructField('high_2', DoubleType(), True),
                     StructField('high_3', DoubleType(), True),
                     StructField('high_4', DoubleType(), True),
                     StructField('high_5', DoubleType(), True),
                     StructField('high_6', DoubleType(), True),
                     StructField('high_7', DoubleType(), True)])


data = spark.read.csv("Data/HIGGS.csv.gz",header=False, schema=schema).repartition(10).cache()
print(data.rdd.getNumPartitions())

feature_names = ["low_1","low_2","low_3","low_4","low_5","low_6"
                 ,"low_7","low_8","low_9","low_10","low_11","low_12"
                 ,"low_13","low_14","low_15","low_16","low_17","low_18"
                 ,"low_19","low_20","low_21","high_1","high_2","high_3"
                 ,"high_4","high_5","high_6","high_7"]  



data_25 = data.sample(False,0.25, 91)
(train,test) = data_25.randomSplit([0.8,0.2], 666)
train.cache()
test.cache()

# models

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",metricName="accuracy")
dtc_assembler = VectorAssembler(inputCols = feature_names, outputCol = 'features') 
dtc = DecisionTreeClassifier(labelCol="label", featuresCol="features", seed=111)
dtr = DecisionTreeRegressor(featuresCol='features', labelCol='label', predictionCol="bin_prediction")
lr = LogisticRegression(featuresCol='features', labelCol='label')
binarizer = Binarizer(threshold=0.5, inputCol="bin_prediction", outputCol="prediction")


# define pipeline
pipeline = Pipeline(stages=[dtc_assembler])


# 3^2  models = 6
dtc_grid = ParamGridBuilder().baseOn({pipeline.stages:[dtc_assembler, dtc]})\
                            .addGrid(dtc.maxDepth, [5, 10, 15])\
                            .addGrid(dtc.maxBins, [16, 32])\
                            .build()
# 3^2 models = 9
dtr_grid = ParamGridBuilder().baseOn({pipeline.stages:[dtc_assembler, dtr, binarizer]})\
                            .addGrid(dtr.maxDepth, [5, 10, 15])\
                            .addGrid(dtr.maxBins, [6, 16, 32])\
                            .build()
# 2^3 models = 8
lr_grid = ParamGridBuilder().baseOn({pipeline.stages:[dtc_assembler, lr]})\
                             .addGrid(lr.regParam, [0.01, 0.05])\
                             .addGrid(lr.maxIter, [10, 100])\
                             .addGrid(lr.elasticNetParam, [0.0, 0.1])\
                             .build()

paramGrid = dtc_grid + dtr_grid + lr_grid

#fit cv

cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, parallelism=10, numFolds=3, seed=16)
cvModel = cv.fit(train)
predictions = cvModel.transform(test)
accuracy = evaluator.evaluate(predictions)

print(cvModel.avgMetrics)
print("trained models",len(cvModel.avgMetrics))
print("\n")
print("Decision tree classifier models accuracy", cvModel.avgMetrics[:6])
print("\n")
print("Decision tree regression models accuracy", cvModel.avgMetrics[6:15])
print("\n")
print("Logistic regression models accuracy", cvModel.avgMetrics[15:23])

#GET BEST DTC model
print("\n")
print("Best decision tree classifier model")
print("accuracy", cvModel.avgMetrics[np.argmax(cvModel.avgMetrics[:6])])
print(cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics[:6])])

# GET BEST DTR model
print("\n")
print("Best decision tree regression model")
print("accuracy", cvModel.avgMetrics[6 + np.argmax(cvModel.avgMetrics[6:15])])
print(cvModel.getEstimatorParamMaps()[6 + np.argmax(cvModel.avgMetrics[6:15])])

# GET BEST LOGR model
print("\n")
print("Best logistic regression model")
print("accuracy", cvModel.avgMetrics[15 + np.argmax(cvModel.avgMetrics[15:23])])
print(cvModel.getEstimatorParamMaps()[15 + np.argmax(cvModel.avgMetrics[15:23])])
print("\n")


# train final models

dtc_assembler = VectorAssembler(inputCols = feature_names, outputCol = 'features') 
data_new = dtc_assembler.transform(data)
(train_total, test_total) = data_new.randomSplit([0.8, 0.2], 44)
train_total.cache()
test_total.cache()

# train DTC

dtc_total = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=15 , maxBins=32, seed=111)
model = dtc_total.fit(train_total)
predictions = model.transform(test_total)
accuracy = evaluator.evaluate(predictions)
print("Final DTC Accuracy = %g " % accuracy)
print('AUC:', BinaryClassificationMetrics(predictions['label','prediction'].rdd).areaUnderROC)
importance = pd.DataFrame(model.featureImportances.toArray(), columns=["values"])
features_col = pd.Series(feature_names)
importance["features"] = features_col
importance = importance.sort_values(by='values', ascending=False)
importance = importance.reset_index(drop=True)
print('1st feature', importance.features[0], importance.values[0][0])
print('2nd feature', importance.features[1], importance.values[1][0])
print('3rd feature', importance.features[2], importance.values[2][0])


# train DTR

dtr_total = DecisionTreeRegressor(featuresCol='features', labelCol='label', predictionCol="bin_prediction", maxDepth=15, maxBins=16, seed=111)
r_model = dtr_total.fit(train_total)
predictions = r_model.transform(test_total)

binarizer = Binarizer(threshold=0.5, inputCol="bin_prediction", outputCol="prediction")
binarizedDataFrame = binarizer.transform(predictions)

accuracy = evaluator.evaluate(binarizedDataFrame)
print("\n")
print("Final DTR Accuracy = %g " % accuracy)
print('AUC:', BinaryClassificationMetrics(binarizedDataFrame['label','prediction'].rdd).areaUnderROC)
importance = pd.DataFrame(model.featureImportances.toArray(), columns=["values"])
features_col = pd.Series(feature_names)
importance["features"] = features_col
importance = importance.sort_values(by='values', ascending=False)
importance = importance.reset_index(drop=True)
print('1st feature', importance.features[0], importance.values[0][0])
print('2nd feature', importance.features[1], importance.values[1][0])
print('3rd feature', importance.features[2], importance.values[2][0])

# train lr

lr_total = LogisticRegression(featuresCol='features', labelCol='label', regParam=0.01, maxIter=10)
lrModel = lr_total.fit(train_total)
predictions = lrModel.transform(test_total)
accuracy = evaluator.evaluate(predictions)
print("\n")
print("Final LR Accuracy = %g " % accuracy)
print('AUC:', BinaryClassificationMetrics(predictions['label','prediction'].rdd).areaUnderROC)
importance = pd.DataFrame(lrModel.coefficients.toArray(), columns=["values"])
features_col = pd.Series(feature_names)
importance["features"] = features_col
importance = importance.sort_values(by='values', ascending=False)
importance = importance.reset_index(drop=True)
print('1st feature', importance.features[0], importance.values[0][0])
print('2nd feature', importance.features[1], importance.values[1][0])
print('3rd feature', importance.features[2], importance.values[2][0])

end = time.time()
print('tiempo' , end-begin)

sc.stop()


