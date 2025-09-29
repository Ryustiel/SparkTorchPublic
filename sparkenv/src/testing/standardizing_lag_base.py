
import time, kagglehub, torch, pandas, pyspark.sql, pyspark.ml, pyspark.sql.functions as functions

from typing import Dict, List
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DateType, ArrayType

spark = (
    pyspark.sql.SparkSession
    .builder
    .appName("StreamingTestRateSource")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Tells spark to interpret ../07 as year 2007, specific to current dataset
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# Etape 4 : Importer de la même manière les données sur la consommation électrique et les prétraiter proprement (mode script)

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

# ============================= SPECIAL PREPROCESSING LAYER =============================

class StandardizingLagBase(pyspark.ml.util.DefaultParamsReadable, pyspark.ml.util.DefaultParamsWritable):
    partition_col = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "partition_col", "Column to partition data by.")
    order_col = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "order_col", "Column to order the sequence by.")
    sequence_feature_cols = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "sequence_feature_cols", "Columns for sequence features.")
    sequence_length = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "sequence_length", "Length of the sequence.")

class StandardizingLagModel(pyspark.ml.Model, StandardizingLagBase):
    """
    This model stores the learned means and standard deviations
    and uses them to scale the generated sequence columns.
    """
    def __init__(self, partition_col: str, order_col: str, sequence_feature_cols: List[str], 
                 sequence_length: int, feature_stats: Dict[str, Dict[str, float]]):
        super().__init__()
        self.feature_stats = feature_stats or {}
        self._setDefault(partition_col=partition_col, order_col=order_col, sequence_feature_cols=sequence_feature_cols, sequence_length=sequence_length)

    def _transform(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        part_col = self.getOrDefault(self.partition_col)
        ord_col = self.getOrDefault(self.order_col)
        sequence_length = self.getOrDefault(self.sequence_length)
        sequence_feature_cols = self.getOrDefault(self.sequence_feature_cols)

        window = pyspark.sql.Window.partitionBy(part_col).orderBy(ord_col)
        
        new_column_expressions = []
        for i in range(1, sequence_length + 1):

            for col_name in sequence_feature_cols:
                raw_lag_expr = functions.lag(functions.col(col_name), i).over(window)

                feature_lag_col_name = f"{col_name}_lag_{i}"
                mask_lag_col_name = f"{feature_lag_col_name}_mask"

                mask_expr = functions.when(raw_lag_expr.isNull(), 1).otherwise(0).alias(mask_lag_col_name)
                
                feature_expr = (raw_lag_expr - self.feature_stats[col_name]['mean']) / self.feature_stats[col_name]['std']
                feature_expr = functions.round(feature_expr, 4)
                feature_expr = functions.coalesce(feature_expr, functions.lit(0.0)).alias(feature_lag_col_name)

                new_column_expressions.append(feature_expr)
                new_column_expressions.append(mask_expr)

        return df.select(df.columns + new_column_expressions)

class StandardizingLagEstimator(pyspark.ml.Estimator, StandardizingLagBase):
    """
    The Estimator that learns statistics (mean, stddev) from the data
    and returns a Model to perform the transformation.
    """
    def __init__(self, partition_col: str, order_col: str, sequence_feature_cols: List[str], sequence_length: int):
        super().__init__()
        self._setDefault(partition_col=partition_col, order_col=order_col, sequence_feature_cols=sequence_feature_cols, sequence_length=sequence_length)
    
    def _fit(self, df: pyspark.sql.DataFrame) -> StandardizingLagModel:
        sequence_feature_cols = self.getOrDefault(self.sequence_feature_cols)

        agg_expressions = []
        for col_name in sequence_feature_cols:
            agg_expressions.append(functions.mean(col_name).alias(f"mean_{col_name}"))
            agg_expressions.append(functions.stddev(col_name).alias(f"stddev_{col_name}"))

        stats_row = df.agg(*agg_expressions).collect()[0]
        
        feature_stats = {}
        for col_name in sequence_feature_cols:
            mean = stats_row[f"mean_{col_name}"]
            std = stats_row[f"stddev_{col_name}"]
            if std == 0: raise ValueError(f"Standard deviation for column {col_name} is zero, cannot standardize.")
            feature_stats[col_name] = {'mean': mean, 'std': std}

        model = StandardizingLagModel(
            partition_col=self.getOrDefault(self.partition_col),
            order_col=self.getOrDefault(self.order_col),
            sequence_feature_cols=self.getOrDefault(self.sequence_feature_cols),
            sequence_length=self.getOrDefault(self.sequence_length),
            feature_stats=feature_stats
        )
        return self._copyValues(model)

# ============================= SPECIAL PREPROCESSING LAYER =============================

scaler = pyspark.ml.feature.StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withStd=True,
    withMean=False
)

sequential_feature_cols = ["Avg_Temperature_C", "Peak_Hours_Usage_kWh", "Energy_Consumption_kWh"]
sequence_length = 2

sequencer = StandardizingLagEstimator(
    partition_col="Household_ID",
    order_col="Date",
    sequence_feature_cols=sequential_feature_cols,
    sequence_length=2
)

sequence_assembler = pyspark.ml.feature.VectorAssembler(
    inputCols=[
        f"{col}_lag_{i}"
        for i in range(1, sequence_length + 1)
        for col in sequential_feature_cols
    ],
    outputCol="sequence_features"
)

mask_assembler = pyspark.ml.feature.VectorAssembler(
    inputCols=[
        f"{col}_lag_{i}_mask"
        for i in range(1, sequence_length + 1)
        for col in sequential_feature_cols
    ],
    outputCol="mask"
)


df.printSchema()
df.show(5, truncate=False)

pipeline = pyspark.ml.Pipeline(stages=[sequencer, mask_assembler, sequence_assembler])
result = pipeline.fit(df).transform(df)

result.select("Date", "Household_ID", "sequence_features", "mask").show(5, truncate=False)
