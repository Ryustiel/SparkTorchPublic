
import kagglehub, pyspark, pyspark.sql.functions as functions

from components import *

spark = (
    pyspark.sql.SparkSession
    .builder
    .appName("ElectricityConsumptionLSTM")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
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

households = df.select("Household_ID").distinct()
_, validate_id = households.randomSplit([0.9, 0.1], seed=42)
validation = df.join(validate_id, on="Household_ID", how="semi")

model_path = "./models/xgboost_energy_predictor"
loaded_model = pyspark.ml.PipelineModel.load(model_path)

results = loaded_model.transform(validation)

rmse_evaluator = pyspark.ml.evaluation.RegressionEvaluator(labelCol="Energy_Consumption_kWh", predictionCol="prediction", metricName="rmse")
mae_evaluator = pyspark.ml.evaluation.RegressionEvaluator(labelCol="Energy_Consumption_kWh", predictionCol="prediction", metricName="mae")
r2_evaluator = pyspark.ml.evaluation.RegressionEvaluator(labelCol="Energy_Consumption_kWh", predictionCol="prediction", metricName="r2")

rmse, mae, r2 = rmse_evaluator.evaluate(results), mae_evaluator.evaluate(results), r2_evaluator.evaluate(results)

print(f"\nMean Absolute Error (MAE) on test data = {mae} : {mae/df.agg(functions.avg('Energy_Consumption_kWh')).collect()[0][0]*100:.2f}% of average consumption")
print(f"R-squared (R2) on test data = {r2}")
print(f"Root Mean Squared Error (RMSE) on test data = {rmse}\n")
