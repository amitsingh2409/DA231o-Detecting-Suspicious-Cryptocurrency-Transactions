import os
import shutil
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    NaiveBayes,
    GBTClassifier,
    LinearSVC,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def spark_split(df, ratios: list = [0.8, 0.2], target_col: str = "target"):
    pos = df.filter(F.col(target_col) == 1)
    neg = df.filter(F.col(target_col) == 0)

    train_pos, test_pos = pos.randomSplit(ratios, seed=1024)
    train_neg, test_neg = neg.randomSplit(ratios, seed=1024)

    return train_pos.union(train_neg), test_pos.union(test_neg)


def train_model():
    shutil.rmtree("models/best_model", ignore_errors=True)
    shutil.rmtree("mlruns", ignore_errors=True)
    shutil.rmtree("mlartifacts", ignore_errors=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    spark = SparkSession.builder.appName("ModelTrainer").getOrCreate()
    df = spark.read.parquet("data/feature_engineered_data.pq")
    # white as 0 and black as 1
    df = df.withColumn("label", (df["label"] == "white").cast("int"))

    assembler = VectorAssembler(
        inputCols=[
            "year",
            "day",
            "length",
            "weight",
            "count",
            "looped",
            "neighbors",
            "income",
            "num_addresses",
            "day_of_week",
            "length_weight_interaction",
            "income_count_interaction",
            "length_boxcox",
            "income_boxcox",
            "weight_boxcox",
            "count_boxcox",
            "neighbors_boxcox",
        ],
        outputCol="features",
    )
    data = assembler.transform(df)

    algorithms = {
        "LogisticRegression": LogisticRegression(featuresCol="features", labelCol="label"),
        "DecisionTreeClassifier": DecisionTreeClassifier(featuresCol="features", labelCol="label"),
        "RandomForestClassifier": RandomForestClassifier(featuresCol="features", labelCol="label"),
        "NaiveBayes": NaiveBayes(featuresCol="features", labelCol="label"),
        "GBTClassifier": GBTClassifier(featuresCol="features", labelCol="label"),
        "LinearSVC": LinearSVC(featuresCol="features", labelCol="label"),
    }

    best_model = None
    best_auc = float("-inf")

    train, test = spark_split(data, target_col="label")

    for name, algorithm in algorithms.items():
        with mlflow.start_run(run_name=name):
            model = algorithm.fit(train)
            predictions = model.transform(test)
            evaluator = BinaryClassificationEvaluator(
                labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
            )
            auc = evaluator.evaluate(predictions)

            # Additional evaluators for other metrics
            precision_evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
            )
            recall_evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="weightedRecall"
            )
            f1_evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="f1"
            )
            accuracy_evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="accuracy"
            )

            precision = precision_evaluator.evaluate(predictions)
            recall = recall_evaluator.evaluate(predictions)
            f1 = f1_evaluator.evaluate(predictions)
            accuracy = accuracy_evaluator.evaluate(predictions)

            mlflow.log_param("algorithm", name)
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("accuracy", accuracy)

            input_example = train.select("features").limit(5).toPandas()
            input_example["features"] = input_example["features"].apply(
                lambda x: x.toArray().tolist()
            )
            mlflow.spark.log_model(model, "model", input_example=input_example)

            if auc > best_auc:
                best_auc = auc
                best_model = model

    best_predictions = best_model.transform(test)
    shutil.rmtree("data/model_predictions.pq", ignore_errors=True)
    best_predictions.write.parquet("data/model_predictions.pq")
    best_model.save("models/best_model")
    spark.stop()
