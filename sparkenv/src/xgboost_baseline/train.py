
import kagglehub, torch, xgboost.spark, pyspark.sql, pyspark.ml, pyspark.sql.functions as functions

from typing import Dict, List
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DateType, ArrayType

from components import ScaledLagTransform, SimpleScalerEstimator, CollapseMaskTransform

spark = (
    pyspark.sql.SparkSession
    .builder
    .appName("ElectricityConsumptionLSTM")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Tells spark to interpret ../07 as year 2007, specific to current dataset
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

file_path = kagglehub.dataset_download(
    handle="samxsam/household-energy-consumption",
    path="household_energy_consumption.csv",
)

df = spark.read.csv(
    file_path,
    header=True,
    schema = StructType([
        StructField("Household_ID", StringType(), nullable=False),
        StructField("Date", DateType(), nullable=False),
        StructField("Energy_Consumption_kWh", DoubleType(), nullable=False),
        StructField("Household_Size", IntegerType(), nullable=False),
        StructField("Avg_Temperature_C", DoubleType(), nullable=False),
        StructField("Has_AC", StringType(), nullable=False), 
        StructField("Peak_Hours_Usage_kWh", DoubleType(), nullable=False),
    ]),
).withColumn("Has_AC", functions.when(functions.col("Has_AC") == "Yes", True).otherwise(False))

# ============================= PIPELINE =============================

sequential_feature_cols = ["Avg_Temperature_C", "Peak_Hours_Usage_kWh", "Energy_Consumption_kWh"]
sequence_length = 5

sequencer = ScaledLagTransform(
    partition_col="Household_ID",
    order_col="Date",
    sequence_feature_cols=sequential_feature_cols,
    sequence_length=sequence_length
)

static_features_scaler = SimpleScalerEstimator(
    inputCols=["Household_Size", "Avg_Temperature_C"],
)

# XGBoost PIPELINE

collapser = CollapseMaskTransform(
    inputCols=[
        f"{col}_lag_{i}"
        for i in range(1, sequence_length + 1)
        for col in sequential_feature_cols
    ]
)

xgboost_assembler = pyspark.ml.feature.VectorAssembler(
    inputCols=[
        f"{col}_lag_{i}"
        for i in range(1, sequence_length + 1)
        for col in sequential_feature_cols
    ] + [
        "Household_Size_scaled",
        "Avg_Temperature_C_scaled",
        "Has_AC"
    ],
    outputCol="features",
    handleInvalid="keep"  # Handle NaNs by keeping them (XGBoost can handle NaNs internally)
)

xgboost_regressor = xgboost.spark.SparkXGBRegressor(
    features_col="features",
    label_col="Energy_Consumption_kWh"
)

# LSTM PIPELINE

sequence_assembler = pyspark.ml.feature.VectorAssembler(
    inputCols=[
        f"{col}_lag_{i}"
        for i in range(1, sequence_length + 1)
        for col in sequential_feature_cols
    ] + [
        f"{col}_lag_{i}_mask"
        for i in range(1, sequence_length + 1)
        for col in sequential_feature_cols
    ],
    outputCol="features"
)

static_features_assembler = pyspark.ml.feature.VectorAssembler(
    inputCols=[
        "Household_Size_scaled",
        "Avg_Temperature_C_scaled",
        "Has_AC"
    ],
    outputCol="static_features"
)

# PIPELINE BUILD

xgboost_pipeline = pyspark.ml.Pipeline(stages=[sequencer, static_features_scaler, collapser, xgboost_assembler, xgboost_regressor])
lstm_pipeline = pyspark.ml.Pipeline(stages=[sequencer, static_features_scaler, sequence_assembler, static_features_assembler])

# ============================= TRAINING =============================

df.printSchema()
df.show(5, truncate=False)

households = df.select("Household_ID").distinct()
train_id, validate_id = households.randomSplit([0.9, 0.1], seed=42)
train, validate = df.join(train_id, on="Household_ID", how="semi"), df.join(validate_id, on="Household_ID", how="semi")

# Classic Single Shot Training

model = xgboost_pipeline.fit(train)
model.write().overwrite().save("./models/xgboost_energy_predictor")

# Cross-Validation Test for tuning hyper parameters
# (this is supposed to help tune hyperparameters but we have none here)

rmse_evaluator = pyspark.ml.evaluation.RegressionEvaluator(
    labelCol="Energy_Consumption_kWh", 
    predictionCol="prediction", 
    metricName="rmse"
)

cross_validator = pyspark.ml.tuning.CrossValidator(
    estimator=xgboost_pipeline,
    estimatorParamMaps=pyspark.ml.tuning.ParamGridBuilder().build(),
    evaluator=rmse_evaluator,
    numFolds=3,
    parallelism=2,
    seed=42
)
cross_validation_result = cross_validator.fit(train)

result = cross_validation_result.bestModel.transform(validate)
result.select("Date", "Energy_Consumption_kWh", "prediction").show(5, truncate=False)

# ============================= EVALUATION ============================

rmse_evaluator = pyspark.ml.evaluation.RegressionEvaluator(labelCol="Energy_Consumption_kWh", predictionCol="prediction", metricName="rmse")
mae_evaluator = pyspark.ml.evaluation.RegressionEvaluator(labelCol="Energy_Consumption_kWh", predictionCol="prediction", metricName="mae")
r2_evaluator = pyspark.ml.evaluation.RegressionEvaluator(labelCol="Energy_Consumption_kWh", predictionCol="prediction", metricName="r2")

rmse, mae, r2 = rmse_evaluator.evaluate(result), mae_evaluator.evaluate(result), r2_evaluator.evaluate(result)

print(f"\nMean Absolute Error (MAE) on test data = {mae} : {mae/df.agg(functions.avg('Energy_Consumption_kWh')).collect()[0][0]*100:.2f}% of average consumption")
print(f"R-squared (R2) on test data = {r2}")
print(f"Root Mean Squared Error (RMSE) on test data = {rmse}\n")

spark.stop()
