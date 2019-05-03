import pandas as pd
import matplotlib.pyplot as plt
import pyspark	
from pyspark.sql.types import StructField, DoubleType, StructType, IntegerType, StringType, NumericType
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.feature import VectorAssembler, Binarizer, StringIndexer, OneHotEncoderEstimator, PCA
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.regression import GeneralizedLinearRegression, LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
import time
import numpy as np


begin = time.time()


# create session

spark = SparkSession.builder \
    .master("local[10]") \
    .appName("COM6012 Assignment 2") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# schema

schema = StructType([StructField('Row_ID', IntegerType(), True),
                     StructField('Household_ID', IntegerType(), True),
                     StructField('Vehiclhe', IntegerType(), True),
                     StructField('Calendar_Year', StringType(), True),
                     StructField('Model_Year', IntegerType(), True),
                     StructField('Blind_Make', StringType(), True),
                     StructField('Blind_Model', StringType(), True),
                     StructField('Blind_Submodel', StringType(), True),
                     StructField('Cat1', StringType(), True),
                     StructField('Cat2', StringType(), True),
                     StructField('Cat3', StringType(), True),
                     StructField('Cat4', StringType(), True),
                     StructField('Cat5', StringType(), True),
                     StructField('Cat6', StringType(), True),
                     StructField('Cat7', StringType(), True),
                     StructField('Cat8', StringType(), True),
                     StructField('Cat9', StringType(), True),
                     StructField('Cat10', StringType(), True),
                     StructField('Cat11', StringType(), True),
                     StructField('Cat12', StringType(), True),
                     StructField('OrdCat', StringType(), True),
                     StructField('Var1', DoubleType(), True),
                     StructField('Var2', DoubleType(), True),
                     StructField('Var3', DoubleType(), True),
                     StructField('Var4', DoubleType(), True),
                     StructField('Var5', DoubleType(), True),
                     StructField('Var6', DoubleType(), True),
                     StructField('Var7', DoubleType(), True),
                     StructField('Var8', DoubleType(), True),
                     StructField('NVCat', StringType(), True),
                     StructField('NVVar1', DoubleType(), True),
                     StructField('NVVar2', DoubleType(), True),
                     StructField('NVVar3', DoubleType(), True),
                     StructField('NVVar4', DoubleType(), True),
                     StructField('label', DoubleType(), True)])

# read file

data = spark.read.csv("Data/ClaimPredictionChallenge/train_set.csv",header=True, schema=schema, nullValue='?').repartition(10).cache()

# preprocessing

clean = data.dropna(how='any')

cat_input = ['Blind_Make','Model_Year',"Cat1","Cat2","Cat3","Cat4","Cat5","Cat6","Cat7","Cat8","Cat9","Cat10","Cat11","Cat12","OrdCat","NVCat"]

cat_output = [column+"_index" for column in cat_input]

cat_hot = [column+"_hot" for column in cat_output]

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(clean) for column in cat_input]
pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(clean).transform(clean)

encoder = OneHotEncoderEstimator(inputCols=cat_output, outputCols=cat_hot)
hot_data = encoder.fit(df_r).transform(df_r)
df = hot_data.select('Blind_Make_index_hot','Model_Year_index_hot','Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8',
                     'NVVar1','NVVar2','NVVar3', 'NVVar4', 'Cat11_index_hot','Cat9_index_hot',
                     'Cat7_index_hot', 'NVCat_index_hot', 'Cat4_index_hot', 'Cat12_index_hot',
                     'Cat1_index_hot', 'Cat5_index_hot','Cat3_index_hot', 'Cat8_index_hot',
                     'Cat2_index_hot', 'Cat6_index_hot','Cat10_index_hot', 'OrdCat_index_hot','label', 'Calendar_Year')

assembler = VectorAssembler( inputCols=['Blind_Make_index_hot','Model_Year_index_hot','Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8',
							 'NVVar1','NVVar2','NVVar3', 'NVVar4', 'Cat11_index_hot','Cat9_index_hot',
							 'Cat7_index_hot', 'NVCat_index_hot', 'Cat4_index_hot', 'Cat12_index_hot',
							 'Cat1_index_hot', 'Cat5_index_hot','Cat3_index_hot', 'Cat8_index_hot',
							 'Cat2_index_hot', 'Cat6_index_hot','Cat10_index_hot', 'OrdCat_index_hot'],
							  outputCol="features")

output = assembler.transform(df)
assembled = output.select("label", "features", 'Calendar_Year')
pca = PCA(k=25, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(assembled)
final = model.transform(assembled).select("pcaFeatures", "label", 'Calendar_Year')

print("explained variance total", sum(model.explainedVariance))

## only positives

sample = final.filter(final.label > 0)
pos_train = sample.filter(sample.Calendar_Year < 2007)
pos_test = sample.filter(sample.Calendar_Year == 2007)
print("positives train", pos_train.count())
print("positives test", pos_test.count())
pos_train.cache()
pos_test.cache()



## full dataset

train, test = final.randomSplit([0.75,0.25], 666)
print("full train",train.count())
print("full test", test.count())
train.cache()
test.cache()

## Undersampling

final_1 = final.filter(final.label > 0)
final_0 = final.filter(final.label == 0)
ratio = final_1.count() / final.count()
sample_0 = final_0.sample(False, ratio , 12)
sampled_df = final_1.union(sample_0)
u_train, u_test = sampled_df.randomSplit([0.75,0.25], 666)
print("Undersampling train", u_train.count())
print("Undersampling test", u_test.count())
u_train.cache()
u_test.cache()


## Oversampling

final_1 = final.filter(final.label > 0)
final_0 = final.filter(final.label == 0)
overratio = final.count() / final_1.count()
over = final_1.sample(True, overratio, seed=111)
sampled_df = final_0.union(over)
o_train, o_test = sampled_df.randomSplit([0.75,0.25], 666)
print("Oversampling train", o_train.count())
print("Oversampling test", o_test.count())
o_train.cache()
o_test.cache()


## linear regression only with labels > 0 


lr = LinearRegression(featuresCol='pcaFeatures', fitIntercept=True, regParam=0.01)
eval = RegressionEvaluator( labelCol="label", predictionCol="prediction", metricName="rmse")


print("Evaluating full dataset:")

lrModel = lr.fit(train)
prediction = lrModel.transform(test)
rmse = eval.evaluate(prediction)
print("RMSE: %.3f" % rmse)
r2 = eval.evaluate(prediction, {eval.metricName: "r2"})
print("r2: %.3f" %r2)

print("Evaluating positives dataset:")

lrModel = lr.fit(pos_train)
prediction = lrModel.transform(pos_test)
rmse = eval.evaluate(prediction)
print("RMSE: %.3f" % rmse)
r2 = eval.evaluate(prediction, {eval.metricName: "r2"})
print("r2: %.3f" %r2)

print("Evaluating Undersampling dataset:")

lrModel = lr.fit(u_train)
prediction = lrModel.transform(u_test)
rmse = eval.evaluate(prediction)
print("RMSE: %.3f" % rmse)
r2 = eval.evaluate(prediction, {eval.metricName: "r2"})
print("r2: %.3f" %r2)

print("Evaluating Oversampling dataset:")

lrModel = lr.fit(o_train)
prediction = lrModel.transform(o_test)
rmse = eval.evaluate(prediction)
print("RMSE: %.3f" % rmse)
r2 = eval.evaluate(prediction, {eval.metricName: "r2"})
print("r2: %.3f" %r2)


## Random Forest

   
## oversampling

print("Evaluating oversampling dataset Random Forest:")
binarizer = Binarizer(threshold=0, inputCol="label", outputCol="bin_labels")
train_bin = binarizer.transform(o_train)
test_bin = binarizer.transform(o_test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
dt = RandomForestClassifier(labelCol="bin_labels", featuresCol="pcaFeatures", numTrees=10, maxDepth=30)
dtModel = dt.fit(train_bin)
predictions = dtModel.transform(test_bin)
accuracy = evaluator.evaluate(predictions)
print("LR Accuracy = %g " % accuracy)
print('AUC:', BinaryClassificationMetrics(predictions['label','prediction'].rdd).areaUnderROC)


## gamma regression with predictions

gam = predictions.filter(predictions.prediction > 0).filter(predictions.label > 0)

glr = GeneralizedLinearRegression(labelCol="label", featuresCol="pcaFeatures", predictionCol="gammaprediction",family="gamma", link="Inverse", maxIter=10)

## Fit the model

model = glr.fit(gam)
gammapred = model.transform(gam)

evaluator = RegressionEvaluator(labelCol="label", predictionCol="gammaprediction", metricName="r2")
r2 = evaluator.evaluate(gammapred)
print("Evaluating gamma prediction :")
print("R2 = %g " % r2)


end = time.time()
print('tiempo' , end-begin)

sc.stop()


