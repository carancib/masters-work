import pandas as pd
import matplotlib.pyplot as plt
import pyspark	
from pyspark.sql.types import StructField, DoubleType, StructType, IntegerType, StringType, NumericType
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Binarizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.linalg import Vectors
import time
import numpy as np
from pyspark.ml.feature import PCA
from pyspark.ml.stat import Correlation
import seaborn as sns


# create session

spark = SparkSession.builder \
    .master("local[8]") \
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

data = spark.read.csv("Data/ClaimPredictionChallenge/train_set.csv",header=True, schema=schema, nullValue='?').cache()

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


assembler = VectorAssembler(
    inputCols=['Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8',
 'NVVar1','NVVar2','NVVar3', 'NVVar4','label']
    , outputCol="total")

output = assembler.transform(df)
assembled = output.select("total")


# convert to vector column first
vector_col = "total"

# get correlation matrix
matrix = Correlation.corr(assembled, vector_col).head()
print("Pearson correlation matrix:\n" + str(matrix[0]))


m = matrix[0]
rows = m.toArray().tolist()
df = spark.createDataFrame(rows, ['Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8',
 'NVVar1','NVVar2','NVVar3', 'NVVar4','label'])

pan = df.toPandas()

corr = pan
plt.figure(figsize = (12,8))
plot = sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           annot=True,
           )
plot.figure.savefig("Q2_heatmap.png")

# ## calculate gini

# def gini(actual, pred):
#     assert (len(actual) == len(pred))
#     all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
#     all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
#     totalLosses = all[:, 0].sum()
#     giniSum = all[:, 0].cumsum().sum() / totalLosses

#     giniSum -= (len(actual) + 1) / 2.
#     return giniSum / len(actual)


# def gini_normalized(actual, pred):
#     return gini(actual, pred) / gini(actual, actual)

# actual = np.array(prediction.select("label").cache().collect())
# predictions = np.array(prediction.select("prediction").cache().collect())

# gini_predictions = gini(actual, predictions)
# gini_max = gini(actual, actual)
# ngini= gini_normalized(actual, predictions)
# print('Gini: %.3f, Max. Gini: %.3f, Normalized Gini: %.3f' % (gini_predictions, gini_max, ngini))

# ## log reg

# binarizer = Binarizer(threshold=0, inputCol="label", outputCol="bin_labels")
# train_bin = binarizer.transform(train)
# test_bin = binarizer.transform(test)
# evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
# dt = LogisticRegression(featuresCol= 'pcaFeatures', labelCol='bin_labels')
# dtModel = dt.fit(train_bin)
# predictions = dtModel.transform(test_bin)
# accuracy = evaluator.evaluate(predictions)
# print("LR Accuracy = %g " % accuracy)
# print('AUC:', BinaryClassificationMetrics(predictions['label','prediction'].rdd).areaUnderROC)


