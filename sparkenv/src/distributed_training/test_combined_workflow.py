"""
Combined Spark and Torch workflow for training using the TorchDistributor.
"""

# 2. Import the training and validation data and create the downloaders.

# 3. Figure out how to use the TorchDistributor to distribute the training.

import kagglehub, pathlib, functools
import pyspark.sql, pyspark.ml, pyspark.sql.functions as functions
import torch, pandas as pd, numpy as np

from typing import Dict, List
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DateType, ArrayType

from components import SequencerEstimator, SimpleScalerEstimator, CustomVectorAssembler
from train import LSTMRegressor, ParquetIterableDataset, spark_to_model_input

spark = (
    pyspark.sql.SparkSession
    .builder
    .appName("ElectricityConsumptionLSTM")
    .config("spark.sql.shuffle.partitions", "2")  # To test multi parquet files.
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Tells spark to interpret ../07 as year 2007, specific to current dataset
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

file_path = kagglehub.dataset_download(
    handle="samxsam/household-energy-consumption",
    path="household_energy_consumption.csv",
)

target_folder = pathlib.Path(__file__).parent / "data"

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

# Layers

scaler = SimpleScalerEstimator(
    inputCols=["Energy_Consumption_kWh", "Household_Size", "Avg_Temperature_C"],
)
sequencer = SequencerEstimator(  # Create a _sequence and a _mask vector column for each input feature, scale them
    partition_col="Household_ID",
    order_col="Date",
    sequence_features_cols=["Avg_Temperature_C", "Peak_Hours_Usage_kWh", "Energy_Consumption_kWh"],
    sequence_length=5,
)
aggregator = CustomVectorAssembler(
    inputCols=["Household_Size_scaled", "Avg_Temperature_C_scaled", "Has_AC"],
    outputCol="static_features",
)

pipeline = pyspark.ml.Pipeline(stages=[scaler, sequencer])

pipeline_model = pipeline.fit(df)
df = pipeline_model.transform(df)

households = df.select("Household_ID").distinct()
train_id, validate_id = households.randomSplit([0.9, 0.1], seed=42)
train_df, validate_df = df.join(train_id, on="Household_ID", how="semi"), df.join(validate_id, on="Household_ID", how="semi")

train_df.orderBy(functions.rand()).write.mode("overwrite").parquet(str(target_folder / "energy_data_train"))
validate_df.write.mode("overwrite").parquet(str(target_folder / "energy_data_validate"))

# ============================= TRAIN =============================

spark_preprocess_transform = functools.partial(
        spark_to_model_input,
        sequence_features=["Avg_Temperature_C", "Peak_Hours_Usage_kWh", "Energy_Consumption_kWh"],
        static_features=["Household_Size_scaled", "Avg_Temperature_C_scaled", "Has_AC"],
        target="Energy_Consumption_kWh_scaled"
    )

def training_loop():
    
    train_dataset = ParquetIterableDataset(
        file_path=str(target_folder / "energy_data_train"),
        batch_size=20,
        cur_shard=0,
        num_shards=1,
        transform_func=spark_preprocess_transform
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    model = LSTMRegressor(seq_input_size=3, still_input_size=3, still_hidden_size=64, seq_hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    num_epochs = 5
    for epoch in range(num_epochs):
        
        model.train()  # Put model in training mode
        total_train_loss = 0
        
        for data in train_loader:
            optimizer.zero_grad()

            predictions = model(seq_x=data["sequences"], seq_mask=data["masks"], still_x=data["statics"])

            loss = criterion(predictions, data["targets"])
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # TODO MAYBE : Early stopping, gradient clipping, lr scheduler, etc.
            
        avg_train_loss = total_train_loss / train_dataset.num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

    # TODO : Distribute this
    torch.save(model.state_dict(), str(target_folder / "lstm_model.pth"))

training_loop()

# ============================ EVAL =============================

validation_dataset = ParquetIterableDataset(
        file_path=str(target_folder / "energy_data_validate"),
        batch_size=20,
        transform_func=spark_preprocess_transform
    )
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=None)

model = LSTMRegressor(seq_input_size=3, still_input_size=3, still_hidden_size=64, seq_hidden_size=64)
model.load_state_dict(torch.load(target_folder / "lstm_model.pth"))
criterion = torch.nn.MSELoss()

model.eval() # Put the model in evaluation mode (disables dropout, etc.)
total_val_loss = 0
with torch.no_grad():
    for data in val_loader:
        predictions = model(seq_x=data["sequences"], seq_mask=data["masks"], still_x=data["statics"])
        loss = criterion(predictions, data["targets"])
        total_val_loss += loss.item()

avg_val_loss = total_val_loss / validation_dataset.num_batches

print(f"Val Loss: {avg_val_loss:.4f}")
