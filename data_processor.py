import shutil
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from scipy.stats import boxcox


def process_data():
    spark = SparkSession.builder.appName("DataProcessor").getOrCreate()
    df = spark.read.parquet("data/bitcoin.pq")
    processed_df = df.dropna()

    # Down-sample the majority class
    majority_class = processed_df.filter(F.col("label") == "white")
    minority_class = processed_df.filter(F.col("label") != "white")
    # change minority class label to "black"
    minority_class = minority_class.withColumn("label", F.lit("black"))
    # minority class has around 42k rows
    majority_class_downsampled = majority_class.sample(fraction=0.015, seed=1024)
    balanced_df = majority_class_downsampled.union(minority_class)

    shutil.rmtree("data/processed_data.pq", ignore_errors=True)
    balanced_df.write.parquet("data/processed_data.pq")
    spark.stop()


def feature_engineering():
    spark = SparkSession.builder.appName("EDAFeatureEngineering").getOrCreate()
    df = spark.read.parquet("data/processed_data.pq")

    # New features
    df = df.withColumn("num_addresses", F.count("address").over(Window.partitionBy("address")))
    df = df.withColumn(
        "day_of_week",
        F.dayofweek(F.to_date(F.concat_ws("-", F.col("year"), F.col("day")), "yyyy-D")),
    )
    df = df.withColumn("is_holiday", F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0))

    # Interaction features
    df = df.withColumn("length_weight_interaction", F.col("length") * F.col("weight"))
    df = df.withColumn("income_count_interaction", F.col("income") * F.col("count"))

    # Boxcox transformation
    pandas_df = df.toPandas()
    for feature in ["length", "income", "weight", "count", "neighbors"]:
        pandas_df[f"{feature}_boxcox"], _ = boxcox(pandas_df[feature] + 1)

    # Convert back to Spark DataFrame
    df = spark.createDataFrame(pandas_df)
