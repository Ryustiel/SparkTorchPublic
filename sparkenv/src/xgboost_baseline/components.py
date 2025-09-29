
import pyspark.sql, pyspark.ml, pyspark.sql.functions as functions

from typing import Dict, List, Optional
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DateType, ArrayType

# ============================= SCALED LAG =============================

class ScaledLagBase(pyspark.ml.util.DefaultParamsReadable, pyspark.ml.util.DefaultParamsWritable):
    partition_col = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "partition_col", "Column to partition data by.")
    order_col = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "order_col", "Column to order the sequence by.")
    sequence_feature_cols = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "sequence_feature_cols", "Columns for sequence features.")
    sequence_length = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "sequence_length", "Length of the sequence.")
    no_scaling_cols = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "no_scaling_cols", "Columns to skip scaling.")

class ScaledLagModel(pyspark.ml.Model, ScaledLagBase):
    """
    This model stores the learned means and standard deviations
    and uses them to scale the generated sequence columns.
    """
    feature_stats = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "feature_stats", "Dictionary of feature statistics for scaling.")

    def __init__(self, feature_stats: Optional[Dict[str, Dict[str, float]]] = None, **kwargs):
        super().__init__()
        if feature_stats: self._setDefault(feature_stats=feature_stats)
        if kwargs: self._setDefault(**kwargs)

    def _transform(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        part_col = self.getOrDefault(self.partition_col)
        ord_col = self.getOrDefault(self.order_col)
        sequence_length = self.getOrDefault(self.sequence_length)
        sequence_feature_cols = self.getOrDefault(self.sequence_feature_cols)
        no_scaling_cols = self.getOrDefault(self.no_scaling_cols)
        feature_stats = self.getOrDefault(self.feature_stats)

        window = pyspark.sql.Window.partitionBy(part_col).orderBy(ord_col)
        
        new_column_expressions = []
        for i in range(1, sequence_length + 1):

            for col_name in sequence_feature_cols:
                raw_lag_expr = functions.lag(functions.col(col_name), i).over(window)

                feature_lag_col_name = f"{col_name}_lag_{i}"
                mask_lag_col_name = f"{feature_lag_col_name}_mask"

                mask_expr = functions.when(raw_lag_expr.isNull(), 1).otherwise(0).alias(mask_lag_col_name)
                
                if col_name in no_scaling_cols:
                    feature_expr = raw_lag_expr
                else:
                    feature_expr = (raw_lag_expr - feature_stats[col_name]['mean']) / feature_stats[col_name]['std']
                    feature_expr = functions.round(feature_expr, 4)
                feature_expr = functions.coalesce(feature_expr, functions.lit(0.0)).alias(feature_lag_col_name)

                new_column_expressions.append(feature_expr)
                new_column_expressions.append(mask_expr)

        return df.select(df.columns + new_column_expressions)

class ScaledLagTransform(pyspark.ml.Estimator, ScaledLagBase):
    """Lag the specified columns and scale them using statistics from the original columns."""
    def __init__(self, partition_col: str, order_col: str, sequence_feature_cols: List[str], sequence_length: int, no_scaling_cols: List[str] = []):
        super().__init__()
        self._setDefault(partition_col=partition_col, order_col=order_col, sequence_feature_cols=sequence_feature_cols, sequence_length=sequence_length, no_scaling_cols=no_scaling_cols)
    
    def _fit(self, df: pyspark.sql.DataFrame) -> ScaledLagModel:
        sequence_feature_cols = self.getOrDefault(self.sequence_feature_cols)
        no_scaling_cols = self.getOrDefault(self.no_scaling_cols)

        agg_expressions = []
        for col_name in sequence_feature_cols:
            if col_name in no_scaling_cols:
                continue  # Skip mean calculation for no_scaling_cols
            agg_expressions.append(functions.mean(col_name).alias(f"mean_{col_name}"))
            agg_expressions.append(functions.stddev(col_name).alias(f"stddev_{col_name}"))

        stats_row = df.agg(*agg_expressions).collect()[0]
        
        feature_stats = {}
        for col_name in sequence_feature_cols:
            if col_name in no_scaling_cols:
                continue
            mean = stats_row[f"mean_{col_name}"]
            std = stats_row[f"stddev_{col_name}"]
            if std == 0: raise ValueError(f"Standard deviation for column {col_name} is zero, cannot standardize.")
            feature_stats[col_name] = {'mean': mean, 'std': std}

        model = ScaledLagModel(feature_stats=feature_stats)
        return self._copyValues(model)

# ============================= COLLAPSE MASK TRANSFORM =============================

class CollapseMaskTransform(pyspark.ml.Transformer, pyspark.ml.util.DefaultParamsReadable, pyspark.ml.util.DefaultParamsWritable):
    """Replaces values in input columns with NaN where the corresponding mask column is 1. Modifies the inputCols in place."""
    inputCols = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "inputCols", "Input columns to transform.")

    def __init__(self, inputCols: List[str] = []):
        super().__init__()
        self._setDefault(inputCols=inputCols)

    def _transform(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        # Go through each input column, look for a column with the _mask suffix and the value of the column into NaN where the mask is 1
        inputCols = self.getOrDefault(self.inputCols)
        for col in inputCols:
            mask_col_name = f"{col}_mask"

            if mask_col_name not in df.columns:
                raise ValueError(f"Mask column {mask_col_name} not found for input column {col}.")

            df = df.withColumn(
                col,
                functions.when(
                    functions.col(mask_col_name) == 1,
                    functions.lit(None)  # Use lit(None) for null, which XGBoost understands
                ).otherwise(
                    functions.col(col)
                )
            )
        return df

# ============================= SIMPLE SCALER =============================

class SimpleScalerBase(pyspark.ml.util.DefaultParamsReadable, pyspark.ml.util.DefaultParamsWritable):
    inputCols = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "inputCols", "Input columns to scale.")

class SimpleScalerModel(pyspark.ml.Model, SimpleScalerBase):
    """
    This model stores the learned means and standard deviations
    and uses them to scale the generated columns.
    """
    feature_stats = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "feature_stats", "Dictionary of feature statistics for scaling.")
    
    def __init__(self, feature_stats: Optional[Dict[str, Dict[str, float]]] = None, **kwargs):
        super().__init__()
        if feature_stats: self._setDefault(feature_stats=feature_stats)
        if kwargs: self._setDefault(**kwargs)

    def _transform(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        inputCols = self.getOrDefault(self.inputCols)
        feature_stats = self.getOrDefault(self.feature_stats)
        for col in inputCols:
            mean = feature_stats[col]['mean']
            std = feature_stats[col]['std']
            if std == 0: raise ValueError(f"Standard deviation for column {col} is zero, cannot standardize.")
            df = df.withColumn(
                f"{col}_scaled",
                (functions.col(col) - mean) / std
            )
        return df

class SimpleScalerEstimator(pyspark.ml.Estimator, SimpleScalerBase):
    """
    Standardize the specified columns (subtract mean, divide by std) 
    and put the results in new columns with a _scaled suffix.
    """
    def __init__(self, inputCols: List[str]):
        super().__init__()
        self._setDefault(inputCols=inputCols)

    def _fit(self, df: pyspark.sql.DataFrame) -> SimpleScalerModel:
        inputCols = self.getOrDefault(self.inputCols)

        agg_expressions = []
        for col_name in inputCols:
            agg_expressions.append(functions.mean(col_name).alias(f"mean_{col_name}"))
            agg_expressions.append(functions.stddev(col_name).alias(f"stddev_{col_name}"))

        stats_row = df.agg(*agg_expressions).collect()[0]
        
        feature_stats = {}
        for col_name in inputCols:
            mean = stats_row[f"mean_{col_name}"]
            std = stats_row[f"stddev_{col_name}"]
            if std == 0: raise ValueError(f"Standard deviation for column {col_name} is zero, cannot standardize.")
            feature_stats[col_name] = {'mean': mean, 'std': std}

        model = SimpleScalerModel(feature_stats=feature_stats)
        return self._copyValues(model)
    