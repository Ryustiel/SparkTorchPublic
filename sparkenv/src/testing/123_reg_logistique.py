
import time, kagglehub, pandas, pyspark.sql, pyspark.ml

from typing import List
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, BooleanType

spark = (
    pyspark.sql.SparkSession
    .builder
    .appName("StreamingTestRateSource")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


def test_stream(rows_per_second: int = 3, processing_time: int = 3, stop_at_batch: int = 5):
    
    rate_df = (
        spark.readStream
        .format("rate")
        .option("rowsPerSecond", rows_per_second)
        .load()
    )
    processed_df = rate_df.withColumn("is_even", rate_df["value"] % 2 == 0)
    
    query = (
        processed_df.writeStream
        .format("console")
        .outputMode("append")
        .trigger(processingTime=f'{processing_time} seconds')
        .start()
    )
    
    while query.isActive:
        last_progress = query.lastProgress
        if last_progress and last_progress['batchId'] >= stop_at_batch:
            print(f"--- Reached target batch ID {last_progress['batchId']}. Stopping query. ---")
            query.stop()
        
        time.sleep(1)  # Check rate

    query.awaitTermination()

def get_from_kaggle(dataset_name: str, internal_path: str = "") -> pandas.DataFrame:
    """Downloads a dataset from kaggle as a pandas DataFrame."""
    df = kagglehub.dataset_load(
        adapter=kagglehub.KaggleDatasetAdapter.PANDAS,
        handle=dataset_name,
        path=internal_path,
    )
    return df


if __name__ == "__main__":
    
    file_path = kagglehub.dataset_download(
        handle="samikshadalvi/pcos-diagnosis-dataset",
        path="pcos_dataset.csv",
    )
    
    df = spark.read.csv(
        file_path,
        header=True,
        schema = StructType([
            StructField("Age", IntegerType(), True),
            StructField("BMI", DoubleType(), True),
            StructField("Menstrual_Irregularity", IntegerType(), True),
            StructField("Testosterone_Level(ng/dL)", DoubleType(), True),
            StructField("Antral_Follicle_Count", IntegerType(), True),
            StructField("PCOS_Diagnosis", IntegerType(), True),
        ]),
    )
    df.printSchema()
    df.show(5, truncate=False)

    # Etape 2 : Preprocess les données

    feature_columns = ["Age", "BMI", "Menstrual_Irregularity", "Testosterone_Level(ng/dL)", "Antral_Follicle_Count"]

    feature_assembler = pyspark.ml.feature.VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )

    scaler = pyspark.ml.feature.StandardScaler(
        inputCol="features",
        outputCol="scaledFeatures",
        withStd=True,  # Scale to unit standard deviation
        withMean=False # Do not shift to zero mean (a common choice)
    )

    # Etape 3 : Régression logistique avec une Pipeline

    logistic_regression = pyspark.ml.classification.LogisticRegression(
        featuresCol="scaledFeatures",
        labelCol="PCOS_Diagnosis",
        predictionCol="prediction",
        rawPredictionCol="rawPrediction",
        probabilityCol="probability",
    )

    pipeline = pyspark.ml.Pipeline(stages=[feature_assembler, scaler, logistic_regression])

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    predictions = pipeline.fit(train).transform(test)

    predictions.select("PCOS_Diagnosis", "prediction", "probability").show(20, truncate=False)

    evaluator = pyspark.ml.evaluation.BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction", 
        labelCol="PCOS_Diagnosis", 
        metricName="areaUnderROC"
    )
    auc = evaluator.evaluate(predictions)
    print(f"AUC: {auc}")

spark.stop()
