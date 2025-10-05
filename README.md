# Spark + PyTorch: Distributed LSTM Training for Time-Series Forecasting

A demonstration of an end-to-end MLOps workflow combining Apache Spark for large-scale data preprocessing and PyTorch for distributed deep learning. This project trains an LSTM model to predict household energy consumption, using Spark's `TorchDistributor` to manage the distributed training environment.

---

## TLDR

-   **Preprocessing at Scale:** A custom Spark ML Pipeline ingests, cleans, and transforms time-series data into sequences suitable for an LSTM.
-   **Distributed Training:** PySpark's `TorchDistributor` launches a PyTorch `DistributedDataParallel` (DDP) training job on the Spark cluster.
-   **Efficient Data Loading:** A custom `ParquetIterableDataset` streams data directly from Parquet files to shard it across worker nodes.
-   **Hybrid Model:** The PyTorch model combines an LSTM for handling temporal sequences (essentially past energy consumption and historical temperature) and a simple feed-forward network for static features (like household size).
-   **Baseline Comparison:** An included XGBoost model trained with Spark ML serves as a baseline.
-   **Containerized & Reproducible:** The entire environment, including Java, Python, and all dependencies, is packaged in a Docker container for one-command execution.

---

## The Problem & Dataset

The goal is to predict daily energy consumption for individual households based on historical usage and other relevant factors. This is a classic time-series forecasting problem with a twist: we have both sequential data (daily metrics) and static data (household attributes).

The project uses this recent [**Household Energy Consumption**](https://www.kaggle.com/datasets/samxsam/household-energy-consumption) dataset from Kaggle.

### Dataset Overview

This dataset presents detailed energy consumption records from various households over one month. With 90,000 rows, it's well-suited for time-series analysis.

| Column Name              | Description                                        | Role in Model         |
| ------------------------ | -------------------------------------------------- | --------------------- |
| `Household_ID`           | Unique identifier for each household               | Partitioning Key      |
| `Date`                   | The date of the energy usage record                | Ordering Key          |
| `Energy_Consumption_kWh` | Total energy consumed by the household in kWh      | **Target** & Sequence |
| `Household_Size`         | Number of individuals living in the household      | Static Feature        |
| `Avg_Temperature_C`      | Average daily temperature in degrees Celsius       | Sequence & Static     |
| `Has_AC`                 | Indicates if the household has air conditioning    | Static Feature        |
| `Peak_Hours_Usage_kWh`   | Energy consumed during peak hours in kWh           | Sequence Feature      |

---

## Modeling & Preprocessing Strategy

### The Model: A Hybrid LSTM

I chose a hybrid architecture to handle the different types of features:
1.  An **LSTM layer** processes the sequential data (`Avg_Temperature_C`, `Peak_Hours_Usage_kWh`, `Energy_Consumption_kWh` from previous days).
2.  A **Feed-Forward Network** processes the static features (`Household_Size`, current day's / forecasted `Avg_Temperature_C`, `Has_AC`).
3.  The outputs from both are **concatenated** and passed through a final linear layer to produce the prediction.

This approach was an attempt to do something a little more efficient than simply passing on the sequence. Additionally, the model uses `torch.nn.utils.rnn.pack_padded_sequence` to ignore unknown time dependent values in the LSTM computation. This is useful for training and infering from early in each household's history where fewer than `sequence_length` data points are available.

*A note on optimality:* While this model is suitable, the primary goal of this project was to explore and implement a robust distributed training workflow with Spark and PyTorch. The architecture could be more in sync with the state of the art or made fancier (like adding attention, more layers), but it serves as a non-trivial example for demonstrating the end-to-end pipeline.

### The Preprocessing Pipeline: Custom Spark ML Transformers

All preprocessing is done within a `pyspark.ml.Pipeline`. The trained pipeline can be saved and restored using native Spark functions and is useful for ensuring consistant training and inference. The key components are custom `Estimator` and `Transformer` classes.

1.  **`SimpleScalerEstimator`**: A standard scaler that computes the mean and standard deviation for specified columns (`inputCols`) on the training data and stores them. The resulting `SimpleScalerModel` applies the normalization to select variables. This is a way to bypass spark's standard VectorScaler which would have had me build and disassemble a vector.
2.  **`SequencerEstimator`**: This is the core of the feature engineering. For each household:
    *   It uses a Spark **window function** partitioned by `Household_ID` and ordered by `Date`.
    *   It collects the last `sequence_length` values for each sequential feature.
    *   It **scales** these values using statistics learned from the entire training set.
    *   It **pads** sequences shorter than `sequence_length` with zeros.
    *   It generates a corresponding **binary mask** (`1` for real data, `0` for padding).
    *   The output for each feature (e.g., `Avg_Temperature_C`) is two new array columns: `Avg_Temperature_C_sequence` and `Avg_Temperature_C_mask`.

The transformed DataFrame is then written to **Parquet** files. This format is highly efficient and preserves the schema for the PyTorch training step.

---

## Tech Stack & Distributed Training Highlights

The most interesting part of this project is the integration of Spark and PyTorch for distributed training.

-   **Frameworks:** Apache Spark `4.0.1`, PySpark, PyTorch `2.8.0`
-   **Data Handling:** Custom Spark ML `Estimator`/`Transformer`, Parquet, PyArrow
-   **Distributed Computing:** `pyspark.ml.torch.distributor.TorchDistributor`, `torch.distributed`, `DistributedDataParallel` (DDP)
-   **Containerization:** Docker with a `python:3.13-slim` base image and OpenJDK 21.
-   **Dependency Management:** `uv`

### The Workflow

1.  **Spark Preprocessing**: The `combined_workflow.py` script starts a Spark session. It runs the ML Pipeline to transform the raw CSV data and saves the result as a sharded Parquet dataset in `./data/energy_data_train`.

2.  **Launch with `TorchDistributor`**:
    ```python
    distributor = pyspark.ml.torch.distributor.TorchDistributor(
        num_processes=2, local_mode=True, use_gpu=False
    )
    distributor.run(training_loop)
    ```
    The `TorchDistributor` takes the custom `training_loop` function and execute it in the specified number of processes.

3.  **Sharded Data Loading in PyTorch**: Inside `training_loop`, each process (or `rank`) initializes its own instance of `ParquetIterableDataset`.
    ```python
    # In ParquetIterableDataset.__init__
    dataset = pyarrow.parquet.ParquetDataset(self.file_path)
    self.shard_fragments = [
        fragment for i, fragment in enumerate(dataset.fragments)
        if i % num_shards == cur_shard
    ]
    ```
    This ensures that each worker process reads a unique subset of the Parquet file fragments. Though the fragment splitting process might not distribute data properly if the number of fragments is not a multiple of the number of processes due to the modulo. The dataset streams data in batches so that it can fit in the RAM.

4.  **Distributed Model Training**:
    ```python
    # In training_loop
    torch.distributed.init_process_group(backend="gloo")
    model = LSTMRegressor(...)
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    ```
    The model is wrapped with `DistributedDataParallel` (DDP) for synchronizing the gradients accross processes during `loss.backward()`. This keeps all model replicas synchronized. `find_unused_parameters=True` was added because the `pack_padded_sequence` step skips the update of some of the weights of the LSTM.

5.  **Synchronization and Saving**: After each epoch, metrics like total loss are manually synchronized using `torch.distributed.all_reduce` to compute a global average. The final model is saved only by the `rank == 0` process to prevent write conflicts.

---

## Installation & Running

The project is designed to be run inside a Docker container, which handles all system-level dependencies like Java and Spark.

**Prerequisites:**
-   Docker

**Build and Run:**

1.  **Clone the repository.**

2.  **Run the container:** You can build the container and run the spark and torch training by calling :
    ```bash
    cd SparkTorchPublic
    docker compose up -d
    docker compose exec sparkenv bash
    spark-submit src/distributed_training/combined_workflow.py
    ```

    If you add kagglehub credentials to the dockerized environment, the dataset will be automatically called via `kagglehub`. The script will then preprocess it, train the model, and save the output artifacts to the `/app/data` directory inside the container.

    If you followed the bash method, you will see output from Spark initialization, followed by the epoch-by-epoch training loss from the PyTorch loop, and finally the evaluation results.

---

## Project Structure

```text
sparkenv/
├── Dockerfile                  # Defines the container environment (Python, Java, Spark, deps)
├── pyproject.toml              # Python dependencies managed by `uv`
├── .python-version             # Specifies Python 3.13
└── src/
    ├── distributed_training/
    │   ├── combined_workflow.py # Main script: Spark preprocessing + distributed Torch training
    │   └── components.py        # Custom Spark ML transformers (SequencerEstimator, SimpleScaler)
    │
    └── xgboost_baseline/
        ├── train.py             # Script to train a baseline XGBoost model using Spark ML
        ├── infer.py             # Script to load and evaluate the saved XGBoost model
        └── components.py        # Custom Spark ML transformers for the XGBoost pipeline
```

