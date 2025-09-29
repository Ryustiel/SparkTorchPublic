
import pyspark.sql, pyspark.ml, pyspark.sql.functions as F

from typing import Dict, List, Optional, TypedDict
from pyspark.sql.types import ArrayType, FloatType

# ============================= SCALED LAG =============================

class FeatureStats(TypedDict):
    mean: float
    std: float

class SequencerBase(pyspark.ml.util.DefaultParamsReadable, pyspark.ml.util.DefaultParamsWritable):
    partition_col = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "partition_col", "Column to partition data by.")
    order_col = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "order_col", "Column to order the sequence by.")
    sequence_length = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "sequence_length", "Length of the sequence.")
    sequence_features_cols = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "sequence_features_cols", "Names of the features that will make up the sequence.")

class SequencerModel(pyspark.ml.Model, SequencerBase):
    """
    Lag the specified columns.
    Retrieve the values from sequence_features_col.
    Create one {name}_sequence and one {name}_mask column for each name in sequence_features_cols.
    """
    feature_stats = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "feature_stats", "Dictionary of feature statistics for scaling.")
    
    def __init__(self, 
                 sequence_length: Optional[int] = 1, 
                 partition_col: Optional[str] = None, 
                 order_col: Optional[str] = None, 
                 sequence_features_col: Optional[str] = None, 
                 sequence_features_cols: Optional[List[str]] = None,
                 feature_stats: Optional[Dict[str, FeatureStats]] = None
                ):
        super().__init__()
        kwargs = {}
            
        for param, value in {
            "partition_col": partition_col, 
            "order_col": order_col, 
            "sequence_length": sequence_length,
            "sequence_features_col": sequence_features_col,
            "sequence_features_cols": sequence_features_cols,
            "feature_stats": feature_stats
        }.items():
            if value is not None:
                kwargs[param] = value
        self._setDefault(**kwargs)

    def _transform(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        partition_col = self.getOrDefault(self.partition_col)
        order_col = self.getOrDefault(self.order_col)
        sequence_length = self.getOrDefault(self.sequence_length)
        sequence_features_cols = self.getOrDefault(self.sequence_features_cols)
        feature_stats = self.getOrDefault(self.feature_stats)
        
        transformed_df = df
        
        window = (
            pyspark.sql.Window
            .partitionBy(partition_col)
            .orderBy(order_col)
            .rowsBetween(-sequence_length, -1)
        )
        
        for feature_name in sequence_features_cols:
            
            scaled_col = f"__{feature_name}_scaled"
            raw_seq_col = f"__{feature_name}_raw_seq"  # Helper columns
            history_len_col = f"__{feature_name}_hist_len"
            padding_len_col = f"__{feature_name}_pad_len"
            
            seq_out_col = f"{feature_name}_sequence"  # Output columns
            mask_out_col = f"{feature_name}_mask"
            
            # Create an array of scaled values
            mean, std = feature_stats[feature_name]['mean'], feature_stats[feature_name]['std']
            transformed_df = transformed_df.withColumn(
                scaled_col,
                F.round((F.col(feature_name) - mean) / std, 4)
            )
            
            transformed_df = transformed_df.withColumn(
                raw_seq_col, 
                F.collect_list(scaled_col).over(window)
            )
            
            # Count the length of the unmasked history for building the mask
            transformed_df = transformed_df.withColumn(
                history_len_col, 
                F.size(F.col(raw_seq_col))
            )
            
            # XXX : Drop rows with no history - this is for pack padded sequence to work
            # transformed_df = transformed_df.filter(F.col(history_len_col) > 0)
            
           # Calculate the padding needed for each row.
            transformed_df = transformed_df.withColumn(
                padding_len_col,
                F.lit(sequence_length) - F.col(history_len_col)
            )

            # Create the feature sequence by concatenating data + padding.
            padding_feature_array = F.array_repeat(F.lit(0.0), F.col(padding_len_col))
            
            transformed_df = transformed_df.withColumn(
                seq_out_col,
                F.concat(F.col(raw_seq_col), padding_feature_array).cast(ArrayType(FloatType()))
            )
            
            # Create the mask by concatenating a sequence of 1s (data) and 0s (padding).
            data_mask_array = F.array_repeat(F.lit(1), F.col(history_len_col))
            padding_mask_array = F.array_repeat(F.lit(0), F.col(padding_len_col))
            
            transformed_df = transformed_df.withColumn(
                mask_out_col,
                F.concat(data_mask_array, padding_mask_array)
            )

            transformed_df = transformed_df.drop(scaled_col, raw_seq_col, history_len_col)
        
        return transformed_df

class SequencerEstimator(pyspark.ml.Estimator, SequencerBase):
    """Estimate and store the variance and mean of the input columns."""
    
    def __init__(self, 
                 sequence_length: Optional[int] = 1, 
                 partition_col: Optional[str] = None, 
                 order_col: Optional[str] = None, 
                 sequence_features_cols: Optional[List[str]] = None
                ):
        super().__init__()
        kwargs = {}
        for param, value in {
            "partition_col": partition_col, 
            "order_col": order_col, 
            "sequence_features_cols": sequence_features_cols,
            "sequence_length": sequence_length
        }.items():
            if value is not None:
                kwargs[param] = value
        self._setDefault(**kwargs)

    def _fit(self, df: pyspark.sql.DataFrame) -> SequencerModel:
        sequence_feature_cols = self.getOrDefault(self.sequence_features_cols)

        agg_expressions = []
        for col_name in sequence_feature_cols:
            agg_expressions.append(F.mean(col_name).alias(f"mean_{col_name}"))
            agg_expressions.append(F.stddev(col_name).alias(f"stddev_{col_name}"))

        stats_row = df.agg(*agg_expressions).collect()[0]
        
        feature_stats = {}
        for col_name in sequence_feature_cols:
            mean = stats_row[f"mean_{col_name}"]
            std = stats_row[f"stddev_{col_name}"]
            if std == 0: raise ValueError(f"Standard deviation for column {col_name} is zero, cannot standardize.")
            feature_stats[col_name] = {'mean': mean, 'std': std}

        model = SequencerModel(feature_stats=feature_stats)
        return self._copyValues(model)


# ============================= SIMPLE SCALER =============================

class SimpleScalerBase(pyspark.ml.util.DefaultParamsReadable, pyspark.ml.util.DefaultParamsWritable):
    inputCols = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "inputCols", "Input columns to scale.")

class SimpleScalerModel(pyspark.ml.Model, SimpleScalerBase):
    """
    This model stores the learned means and standard deviations
    and uses them to scale the generated columns.
    """
    feature_stats = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "feature_stats", "Dictionary of feature statistics for scaling.")
    
    def __init__(self, feature_stats: Optional[Dict[str, FeatureStats]] = None, **kwargs):
        super().__init__()
        if feature_stats: self._setDefault(feature_stats=feature_stats)
        if kwargs: self._setDefault(**kwargs)
        
    def get_params(self, col_name: str) -> FeatureStats:
        """Get the mean and std for a specific column."""
        feature_stats = self.getOrDefault(self.feature_stats)
        if col_name not in feature_stats:
            raise ValueError(f"Column {col_name} not found in feature_stats.")
        return feature_stats[col_name]

    def _transform(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        inputCols = self.getOrDefault(self.inputCols)
        feature_stats = self.getOrDefault(self.feature_stats)
        for col in inputCols:
            mean = feature_stats[col]['mean']
            std = feature_stats[col]['std']
            if std == 0: raise ValueError(f"Standard deviation for column {col} is zero, cannot standardize.")
            df = df.withColumn(
                f"{col}_scaled",
                (F.col(col) - mean) / std
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
            agg_expressions.append(F.mean(col_name).alias(f"mean_{col_name}"))
            agg_expressions.append(F.stddev(col_name).alias(f"stddev_{col_name}"))

        stats_row = df.agg(*agg_expressions).collect()[0]
        
        feature_stats = {}
        for col_name in inputCols:
            mean = stats_row[f"mean_{col_name}"]
            std = stats_row[f"stddev_{col_name}"]
            if std == 0: raise ValueError(f"Standard deviation for column {col_name} is zero, cannot standardize.")
            feature_stats[col_name] = {'mean': mean, 'std': std}

        model = SimpleScalerModel(feature_stats=feature_stats)
        return self._copyValues(model)
    
# ============================= AGGREGATE TO VECTOR =============================
    
class CustomVectorAssembler(pyspark.ml.Transformer, pyspark.ml.util.DefaultParamsReadable, pyspark.ml.util.DefaultParamsWritable):
    """
    Vector Assembler that outputs an ArrayType(FloatType()) column instead of a VectorUDT column.
    This is for saving to parquet with native types.
    """
    inputCols = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "inputCols", "Input columns to aggregate.")
    outputCol = pyspark.ml.param.Param(pyspark.ml.param.Params._dummy(), "outputCol", "Output vector column name.")
    
    def __init__(self, inputCols: Optional[List[str]] = None, outputCol: Optional[str] = None):
        super().__init__()
        kwargs = {}
        for param, value in {
            "inputCols": inputCols, 
            "outputCol": outputCol
        }.items():
            if value is not None:
                kwargs[param] = value
        self._setDefault(**kwargs)

    def _transform(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        inputCols = self.getOrDefault(self.inputCols)
        outputCol = self.getOrDefault(self.outputCol)
        
        assembler = pyspark.ml.feature.VectorAssembler(inputCols=inputCols, outputCol=outputCol)
        df = assembler.transform(df)
        df = df.withColumn(outputCol, F.col(outputCol).cast(ArrayType(FloatType())))
        return df
    