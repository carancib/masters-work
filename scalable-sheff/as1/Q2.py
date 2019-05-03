import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import numpy as np
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator

# create session

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Assignment 1 Q2") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# read file

df = spark.read.csv("Data/ratings.csv", header=True, inferSchema=True)

# create 5 random splits and 5 train-test combinations

splits = df.randomSplit([1.0,1.0,1.0,1.0,1.0], 111)
(training1, test1) = (splits[0].union(splits[1]).union(splits[2]).union(splits[3]), splits[4])
(training2, test2) = (splits[0].union(splits[1]).union(splits[2]).union(splits[4]), splits[3])
(training3, test3) = (splits[0].union(splits[1]).union(splits[4]).union(splits[3]), splits[2])
(training4, test4) = (splits[0].union(splits[4]).union(splits[2]).union(splits[3]), splits[1])
(training5, test5) = (splits[4].union(splits[1]).union(splits[2]).union(splits[3]), splits[0])

# ALS V1

als = ALS(maxIter=10, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")

rmse = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
mae = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")

model1 = als.fit(training1)
model2 = als.fit(training2)
model3 = als.fit(training3)
model4 = als.fit(training4)
model5 = als.fit(training5)

predictions1 = model1.transform(test1)
predictions2 = model2.transform(test2)
predictions3 = model3.transform(test3)
predictions4 = model4.transform(test4)
predictions5 = model5.transform(test5)

rmse1 = rmse.evaluate(predictions1)
mae1 = mae.evaluate(predictions1)
print("RMSE = " + str(rmse1) + " MAE = " +  str(mae1))
rmse2 = rmse.evaluate(predictions2)
mae2 = mae.evaluate(predictions2)
print("RMSE = " + str(rmse2) + " MAE = " +  str(mae2))
rmse3 = rmse.evaluate(predictions3)
mae3 = mae.evaluate(predictions3)
print("RMSE = " + str(rmse3) + " MAE = " +  str(mae3))
rmse4 = rmse.evaluate(predictions4)
mae4 = mae.evaluate(predictions4)
print("RMSE = " + str(rmse4) + " MAE = " +  str(mae4))
rmse5 = rmse.evaluate(predictions5)
mae5 = mae.evaluate(predictions5)
print("RMSE = " + str(rmse5) + " MAE = " +  str(mae5))

total_rmse = (rmse1 , rmse2 , rmse3 , rmse4 , rmse5)
total_mae = (mae1 , mae2 , mae3 , mae4 , mae5)
avg_rmse = np.mean(total_rmse)
avg_mae = np.mean(total_mae)
std_rmse = np.std(total_rmse)
std_mae = np.std(total_mae)
print('model with defaults')
print('avg rmse is = ' + str(avg_rmse))
print('avg mae is = ' + str(avg_mae))
print('std rmse is = ' + str(std_rmse))
print('std mae is = ' + str(std_mae))

# ALS V2

als_v2 = ALS(rank=5, maxIter=10, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")

model6 = als_v2.fit(training1)
model7 = als_v2.fit(training2)
model8 = als_v2.fit(training3)
model9 = als_v2.fit(training4)
model10 = als_v2.fit(training5)

predictions6 = model6.transform(test1)
predictions7 = model7.transform(test2)
predictions8 = model8.transform(test3)
predictions9 = model9.transform(test4)
predictions10 = model10.transform(test5)

rmse6 = rmse.evaluate(predictions6)
mae6 = mae.evaluate(predictions6)
print("RMSE = " + str(rmse6) + " MAE = " +  str(mae6))
rmse7 = rmse.evaluate(predictions7)
mae7 = mae.evaluate(predictions7)
print("RMSE = " + str(rmse7) + " MAE = " +  str(mae7))
rmse8 = rmse.evaluate(predictions8)
mae8 = mae.evaluate(predictions8)
print("RMSE = " + str(rmse8) + " MAE = " +  str(mae8))
rmse9 = rmse.evaluate(predictions9)
mae9 = mae.evaluate(predictions9)
print("RMSE = " + str(rmse9) + " MAE = " +  str(mae9))
rmse10 = rmse.evaluate(predictions10)
mae10 = mae.evaluate(predictions10)
print("RMSE = " + str(rmse10) + " MAE = " +  str(mae10))

total_rmse = (rmse6 , rmse7 , rmse8 , rmse9 , rmse10)
total_mae = (mae6 , mae7 , mae8 , mae9 , mae10)
avg_rmse = np.mean(total_rmse)
avg_mae = np.mean(total_mae)
std_rmse = np.std(total_rmse)
std_mae = np.std(total_mae)
print('model with rank 5')
print('avg rmse is = ' + str(avg_rmse))
print('avg mae is = ' + str(avg_mae))
print('std rmse is = ' + str(std_rmse))
print('std mae is = ' + str(std_mae))

# QUESTION 2 PART C

scores = spark.read.csv("Data/genome-scores.csv", header=True, inferSchema=True)
tags = spark.read.csv("Data/genome-tags.csv", header=True, inferSchema=True)

# Fitting KNN for model 1 , declaring data for other models

to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
m1 = model1.itemFactors
m2 = model2.itemFactors
m3 = model3.itemFactors
m4 = model4.itemFactors
m5 = model5.itemFactors

data1 = m1.select("id", to_vector("features").alias("features"))
data2 = m2.select("id", to_vector("features").alias("features"))
data3 = m3.select("id", to_vector("features").alias("features"))
data4 = m4.select("id", to_vector("features").alias("features"))
data5 = m5.select("id", to_vector("features").alias("features"))

kmeans = KMeans().setK(20).setSeed(11)
knn1 = kmeans.fit(data1)
kpred1 = knn1.transform(data1)

######### FOR MODEL 1

# get top 3 counts clusters

counts = kpred1.groupby('prediction').count().orderBy('count', ascending=False).limit(3)
counts.show()

# get values for top 3 clusters

first = counts.select('prediction').collect()[0]["prediction"]
second = counts.select('prediction').collect()[1]["prediction"]
third = counts.select('prediction').collect()[2]["prediction"]

# filter movies for each cluster

cluster1 = kpred1.filter(kpred1.prediction == first)
cluster2 = kpred1.filter(kpred1.prediction == second)
cluster3 = kpred1.filter(kpred1.prediction == third)

# get movies, tags and relevance for each cluster

movies1 = scores.join(cluster1, [scores.movieId == cluster1.id], how='rightouter')
movies2 = scores.join(cluster2, [scores.movieId == cluster2.id], how='rightouter')
movies3 = scores.join(cluster3, [scores.movieId == cluster3.id], how='rightouter')

# get top 5 tags by sum

top1 = movies1.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)
top2 = movies2.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)
top3 = movies3.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)

#  get top 5 tag labels from genome_tags

tags1 = top1.join(tags, [top1.tagId == tags.tagId], how='left')
tags2 = top2.join(tags, [top2.tagId == tags.tagId], how='left')
tags3 = top3.join(tags, [top3.tagId == tags.tagId], how='left')
print('top tags for model 1')
tags1.show()
tags2.show()
tags3.show()

####### FOR MODEL 2

knn2 = kmeans.fit(data2)
kpred2 = knn2.transform(data2)

# get top 3 counts clusters

counts = kpred2.groupby('prediction').count().orderBy('count', ascending=False).limit(3)
counts.show()

# get values for top 3 clusters

first = counts.select('prediction').collect()[0]["prediction"]
second = counts.select('prediction').collect()[1]["prediction"]
third = counts.select('prediction').collect()[2]["prediction"]

# filter movies for each cluster

cluster1 = kpred2.filter(kpred2.prediction == first)
cluster2 = kpred2.filter(kpred2.prediction == second)
cluster3 = kpred2.filter(kpred2.prediction == third)

# get movies, tags and relevance for each cluster

movies1 = scores.join(cluster1, [scores.movieId == cluster1.id], how='rightouter')
movies2 = scores.join(cluster2, [scores.movieId == cluster2.id], how='rightouter')
movies3 = scores.join(cluster3, [scores.movieId == cluster3.id], how='rightouter')

# get top 5 tags by sum

top1 = movies1.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)
top2 = movies2.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)
top3 = movies3.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)

#  get top 5 tag labels from genome_tags

tags1 = top1.join(tags, [top1.tagId == tags.tagId], how='left')
tags2 = top2.join(tags, [top2.tagId == tags.tagId], how='left')
tags3 = top3.join(tags, [top3.tagId == tags.tagId], how='left')
print('top tags for model 2')
tags1.show()
tags2.show()
tags3.show()

####### FOR MODEL 3

knn3 = kmeans.fit(data3)
kpred3 = knn3.transform(data3)

# get top 3 counts clusters

counts = kpred3.groupby('prediction').count().orderBy('count', ascending=False).limit(3)
counts.show()

# get values for top 3 clusters

first = counts.select('prediction').collect()[0]["prediction"]
second = counts.select('prediction').collect()[1]["prediction"]
third = counts.select('prediction').collect()[2]["prediction"]

# filter movies for each cluster

cluster1 = kpred3.filter(kpred3.prediction == first)
cluster2 = kpred3.filter(kpred3.prediction == second)
cluster3 = kpred3.filter(kpred3.prediction == third)

# get movies, tags and relevance for each cluster

movies1 = scores.join(cluster1, [scores.movieId == cluster1.id], how='rightouter')
movies2 = scores.join(cluster2, [scores.movieId == cluster2.id], how='rightouter')
movies3 = scores.join(cluster3, [scores.movieId == cluster3.id], how='rightouter')

# get top 5 tags by sum

top1 = movies1.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)
top2 = movies2.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)
top3 = movies3.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)

#  get top 5 tag labels from genome_tags

tags1 = top1.join(tags, [top1.tagId == tags.tagId], how='left')
tags2 = top2.join(tags, [top2.tagId == tags.tagId], how='left')
tags3 = top3.join(tags, [top3.tagId == tags.tagId], how='left')
print('top tags for model 3')
tags1.show()
tags2.show()
tags3.show()

####### FOR MODEL 4

knn4 = kmeans.fit(data4)
kpred4 = knn4.transform(data4)

# get top 3 counts clusters

counts = kpred4.groupby('prediction').count().orderBy('count', ascending=False).limit(3)
counts.show()

# get values for top 3 clusters

first = counts.select('prediction').collect()[0]["prediction"]
second = counts.select('prediction').collect()[1]["prediction"]
third = counts.select('prediction').collect()[2]["prediction"]

# filter movies for each cluster

cluster1 = kpred4.filter(kpred4.prediction == first)
cluster2 = kpred4.filter(kpred4.prediction == second)
cluster3 = kpred4.filter(kpred4.prediction == third)

# get movies, tags and relevance for each cluster

movies1 = scores.join(cluster1, [scores.movieId == cluster1.id], how='rightouter')
movies2 = scores.join(cluster2, [scores.movieId == cluster2.id], how='rightouter')
movies3 = scores.join(cluster3, [scores.movieId == cluster3.id], how='rightouter')

# get top 5 tags by sum

top1 = movies1.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)
top2 = movies2.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)
top3 = movies3.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)

#  get top 5 tag labels from genome_tags

tags1 = top1.join(tags, [top1.tagId == tags.tagId], how='left')
tags2 = top2.join(tags, [top2.tagId == tags.tagId], how='left')
tags3 = top3.join(tags, [top3.tagId == tags.tagId], how='left')
print('top tags for model 4')
tags1.show()
tags2.show()
tags3.show()

####### FOR MODEL 5

knn5 = kmeans.fit(data5)
kpred5 = knn5.transform(data5)

# get top 3 counts clusters

counts = kpred5.groupby('prediction').count().orderBy('count', ascending=False).limit(3)
counts.show()

# get values for top 3 clusters

first = counts.select('prediction').collect()[0]["prediction"]
second = counts.select('prediction').collect()[1]["prediction"]
third = counts.select('prediction').collect()[2]["prediction"]

# filter movies for each cluster

cluster1 = kpred5.filter(kpred5.prediction == first)
cluster2 = kpred5.filter(kpred5.prediction == second)
cluster3 = kpred5.filter(kpred5.prediction == third)

# get movies, tags and relevance for each cluster

movies1 = scores.join(cluster1, [scores.movieId == cluster1.id], how='rightouter')
movies2 = scores.join(cluster2, [scores.movieId == cluster2.id], how='rightouter')
movies3 = scores.join(cluster3, [scores.movieId == cluster3.id], how='rightouter')

# get top 5 tags by sum

top1 = movies1.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)
top2 = movies2.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)
top3 = movies3.groupby('tagId').sum('relevance').orderBy('sum(relevance)', ascending=False).limit(5)

#  get top 5 tag labels from genome_tags

tags1 = top1.join(tags, [top1.tagId == tags.tagId], how='left')
tags2 = top2.join(tags, [top2.tagId == tags.tagId], how='left')
tags3 = top3.join(tags, [top3.tagId == tags.tagId], how='left')
print('top tags for model 5')
tags1.show()
tags2.show()
tags3.show()

# stop

spark.stop()
