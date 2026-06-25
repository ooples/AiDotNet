---
title: "M4DatasetLoader<T>"
description: "Loads time series datasets from the M4 Competition for benchmarking forecasting models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.TimeSeries`

Loads time series datasets from the M4 Competition for benchmarking forecasting models.

## For Beginners

The M4 Competition is the gold standard for evaluating time series forecasting models.

**What is M4?**

- A collection of 100,000 real-world time series
- Multiple frequencies: Yearly, Quarterly, Monthly, Weekly, Daily, Hourly
- Standardized train/test splits for fair comparison
- Established benchmark metrics (SMAPE, MASE, OWA)

**Why M4 matters:**

- **Industry standard**: Used by researchers and practitioners worldwide
- **Diverse data**: Business, economic, demographic, and financial series
- **Published baselines**: Compare your model against known benchmarks
- **Academic recognition**: Results published in major forecasting journals

**M4 Dataset Statistics:**

| Frequency | Series Count | Forecast Horizon | Typical History |
|-----------|-------------|------------------|-----------------|
| Yearly | 23,000 | 6 | 13-835 years |
| Quarterly | 24,000 | 8 | 16-866 quarters |
| Monthly | 48,000 | 18 | 42-2794 months |
| Weekly | 359 | 13 | 80-2597 weeks |
| Daily | 4,227 | 14 | 93-9919 days |
| Hourly | 414 | 48 | 700-960 hours |

## How It Works

The M4 Competition (Makridakis Competitions) is a highly influential forecasting competition
that provides 100,000 time series across multiple frequencies for benchmarking forecasting methods.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `M4DatasetLoader(M4Frequency,Int32,String,Boolean)` | Initializes a new instance of the `M4DatasetLoader` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `ForecastHorizon` | Gets the forecast horizon for this frequency. |
| `Frequency` | Gets the frequency of the loaded time series. |
| `Name` |  |
| `SeriesCount` | Gets the number of time series in the dataset. |
| `TestSeries` | Gets the test time series data (ground truth for evaluation). |
| `TotalCount` |  |
| `TrainingSeries` | Gets the training time series data. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildDatasetUrls` | Builds the M4 dataset URLs from a configurable base URL. |
| `CalculateMASE(IReadOnlyList<>,IReadOnlyList<>,IReadOnlyList<>,Int32)` | Calculates the Mean Absolute Scaled Error (MASE) for M4 evaluation. |
| `CalculateSMAPE(IReadOnlyList<>,IReadOnlyList<>)` | Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) for M4 evaluation. |
| `DownloadDatasetAsync(String,CancellationToken)` | Downloads the dataset from the M4 Competition GitHub repository. |
| `DownloadFileAtomicAsync(String,String,String,CancellationToken)` | Downloads a file atomically by writing to a temp file first. |
| `EnsureDataExistsAsync(String,CancellationToken)` | Ensures the dataset files exist locally, downloading if necessary. |
| `GetDefaultDataPath` | Gets the default data path for caching datasets. |
| `GetNextBatch` | Gets the next batch of time series for iteration. |
| `GetSeries(Int32)` | Gets a specific time series by index. |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `OnReset` |  |
| `ParseM4CsvAsync(String,CancellationToken)` | Parses an M4 Competition CSV file. |
| `UnloadDataCore` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DatasetStats` | M4 Competition statistics per frequency. |
| `DatasetUrls` | M4 Competition download URLs. |
| `SharedHttpClient` | Shared HttpClient instance to avoid socket exhaustion. |

