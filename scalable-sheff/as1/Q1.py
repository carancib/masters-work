import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from pyspark.sql.functions import *
from pyspark.sql.functions import split, regexp_extract
import pandas as pd
import matplotlib.pyplot as plt 
from pyspark.sql import functions as sf


# start session

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Assignment 1 Q1") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# read file

df = spark.read.csv("Data/NASA_access_log_Jul95.gz", sep=" ")

df2 = df.drop("_c1", "_c2")
d3 = df2.withColumn('joined_column', 
                    sf.concat(sf.col('_c3'),sf.lit(' '), sf.col('_c4')))
d4 = d3.drop("_c3", "_c4")
old = d4.schema.names
new = ["host", "path", "status", "content size","timestamp"]
d5 = d4.withColumnRenamed('_c0', 'host').withColumnRenamed('_c5', 'path').withColumnRenamed('_c6', 'status').withColumnRenamed('_c7', 'content size').withColumnRenamed('joined_column', 'timestamp')
cols = sf.split(d5['path'], r' ')
df6 = d5.withColumn('path_', cols.getItem(1))
d7 = df6.drop("path")
d7.show(2,truncate = False)

# date format

data = d7.withColumn('fecha', (to_date(d7.timestamp, '[dd/MMM/yyyy:HH:mm:ss Z]'))).withColumn('dia', date_format('fecha', 'E'))
data = data.na.drop(subset=["fecha"])



# count number of request per day, number of different days of week and compute average

counts = data.groupby('dia').count().cache()
counts = counts.withColumnRenamed("count", "amount")
day_count = data.groupBy("dia").agg(countDistinct("fecha").alias("number")).cache()
joined = counts.join(day_count, "dia").cache() 
joined = joined.withColumn("avg", (joined.amount / joined.number))
joined.show()

# convert to pandas for plotting

joinedpd = joined.toPandas()
cats = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
joinedpd['dia'] = pd.Categorical(joinedpd['dia'], categories=cats, ordered=True)
joinedpd = joinedpd.sort_values('dia')
joinedpd = joinedpd.set_index('dia').reindex(cats).reset_index()
plt.style.use('seaborn')
plt.figure()
plt.plot(joinedpd.dia, joinedpd.avg)
plt.savefig('Q1_figA.png', dpi=200, bbox_inches="tight")


# counting gif images

gifs = data.filter(data.path_.like('%.gif%'))
gifcount = gifs.groupby('path_').count().cache().orderBy('count', ascending=False)
gifcount = gifcount.withColumnRenamed("count", "amount")
gifcount.show(20, truncate=False)
gifcountpd = gifcount.toPandas()


# grpah for gif images

gifcountpd = gifcountpd[:20]
plt.figure()
plt.bar(gifcountpd.path_, gifcountpd.amount)
plt.xticks(rotation='vertical')
plt.savefig('Q1_figB.png', dpi=200, bbox_inches="tight")

# stop

spark.stop()