---
title: "Time Series"
description: "All 38 public types in the AiDotNet.timeseries namespace, organized by kind."
section: "API Reference"
---

**38** public types in this namespace, organized by kind.

## Models & Types (34)

| Type | Summary |
|:-----|:--------|
| [`ARIMAModel<T>`](/docs/reference/wiki/timeseries/arimamodel/) | Implements an ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting. |
| [`ARIMAXModel<T>`](/docs/reference/wiki/timeseries/arimaxmodel/) | Implements an ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables) model for time series forecasting. |
| [`ARMAModel<T>`](/docs/reference/wiki/timeseries/armamodel/) | Implements an ARMA (AutoRegressive Moving Average) model for time series forecasting. |
| [`ARModel<T>`](/docs/reference/wiki/timeseries/armodel/) | Implements an AR (AutoRegressive) model for time series forecasting. |
| [`AutoformerModel<T>`](/docs/reference/wiki/timeseries/autoformermodel/) | Implements the Autoformer model for long-term time series forecasting with decomposition. |
| [`BayesianStructuralTimeSeriesModel<T>`](/docs/reference/wiki/timeseries/bayesianstructuraltimeseriesmodel/) | Implements a Bayesian Structural Time Series model for flexible time series forecasting. |
| [`ChronosFoundationModel<T>`](/docs/reference/wiki/timeseries/chronosfoundationmodel/) | Implements the Chronos foundation model for zero-shot time series forecasting. |
| [`DLinearModel<T>`](/docs/reference/wiki/timeseries/dlinearmodel/) | DLinear — decomposition-linear forecaster (Zeng et al., AAAI 2023, "Are Transformers Effective for Time Series Forecasting?"). |
| [`DeepANT<T>`](/docs/reference/wiki/timeseries/deepant/) | Implements DeepANT (Deep Learning for Anomaly Detection in Time Series). |
| [`DeepARModel<T>`](/docs/reference/wiki/timeseries/deeparmodel/) | Implements DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks. |
| [`DynamicRegressionWithARIMAErrors<T>`](/docs/reference/wiki/timeseries/dynamicregressionwitharimaerrors/) | Implements a Dynamic Regression model with ARIMA errors for time series forecasting. |
| [`ExponentialSmoothingModel<T>`](/docs/reference/wiki/timeseries/exponentialsmoothingmodel/) | Represents a model that implements exponential smoothing for time series forecasting. |
| [`GARCHModel<T>`](/docs/reference/wiki/timeseries/garchmodel/) | Represents a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model for time series with changing volatility. |
| [`InformerModel<T>`](/docs/reference/wiki/timeseries/informermodel/) | Implements the Informer model for efficient long-sequence time series forecasting. |
| [`InterventionAnalysisModel<T>`](/docs/reference/wiki/timeseries/interventionanalysismodel/) | Represents a model that analyzes and forecasts time series data with interventions or structural changes. |
| [`LSTMVAE<T>`](/docs/reference/wiki/timeseries/lstmvae/) | Implements LSTM-VAE (Long Short-Term Memory Variational Autoencoder) for anomaly detection. |
| [`MAModel<T>`](/docs/reference/wiki/timeseries/mamodel/) | Implements a Moving Average (MA) model for time series forecasting. |
| [`NBEATSModel<T>`](/docs/reference/wiki/timeseries/nbeatsmodel/) | Implements the N-BEATS (Neural Basis Expansion Analysis for Time Series) model for forecasting. |
| [`NHiTSModel<T>`](/docs/reference/wiki/timeseries/nhitsmodel/) | Implements N-HiTS (Neural Hierarchical Interpolation for Time Series) for efficient long-horizon forecasting. |
| [`NLinearModel<T>`](/docs/reference/wiki/timeseries/nlinearmodel/) | NLinear — normalization-linear forecaster (Zeng et al., AAAI 2023). |
| [`NeuralNetworkARIMAModel<T>`](/docs/reference/wiki/timeseries/neuralnetworkarimamodel/) | Represents a Neural Network ARIMA (Autoregressive Integrated Moving Average) model for time series forecasting. |
| [`ProphetModel<T, TInput, TOutput>`](/docs/reference/wiki/timeseries/prophetmodel/) | Represents a Prophet model for time series forecasting. |
| [`SARIMAModel<T>`](/docs/reference/wiki/timeseries/sarimamodel/) | Implements a Seasonal Autoregressive Integrated Moving Average (SARIMA) model for time series forecasting. |
| [`STLDecomposition<T>`](/docs/reference/wiki/timeseries/stldecomposition/) | Implements Seasonal-Trend decomposition using LOESS (STL) for time series analysis. |
| [`SpectralAnalysisModel<T>`](/docs/reference/wiki/timeseries/spectralanalysismodel/) | Implements spectral analysis for time series data, which transforms time domain signals into the frequency domain. |
| [`StateSpaceModel<T>`](/docs/reference/wiki/timeseries/statespacemodel/) | Implements a State Space Model for time series analysis and forecasting. |
| [`TBATSModel<T>`](/docs/reference/wiki/timeseries/tbatsmodel/) | Implements the TBATS (Trigonometric, Box-Cox transform, ARMA errors, Trend, and Seasonal components) model for complex time series forecasting with multiple seasonal patterns. |
| [`TemporalFusionTransformer<T>`](/docs/reference/wiki/timeseries/temporalfusiontransformer/) | Implements the Temporal Fusion Transformer (TFT) per Lim et al. |
| [`TiDEModel<T>`](/docs/reference/wiki/timeseries/tidemodel/) | TiDE — Time-series Dense Encoder (Das et al., TMLR 2023). |
| [`TimeSeriesIsolationForest<T>`](/docs/reference/wiki/timeseries/timeseriesisolationforest/) | Implements Isolation Forest for time series anomaly detection. |
| [`TransferFunctionModel<T>`](/docs/reference/wiki/timeseries/transferfunctionmodel/) | Implements a Transfer Function Model for time series analysis, which combines ARIMA modeling with external input variables to capture dynamic relationships between multiple time series. |
| [`UnobservedComponentsModel<T, TInput, TOutput>`](/docs/reference/wiki/timeseries/unobservedcomponentsmodel/) | Implements an Unobserved Components Model (UCM) for time series decomposition and forecasting. |
| [`VARMAModel<T>`](/docs/reference/wiki/timeseries/varmamodel/) | Implements a Vector Autoregressive Moving Average (VARMA) model for multivariate time series forecasting. |
| [`VectorAutoRegressionModel<T>`](/docs/reference/wiki/timeseries/vectorautoregressionmodel/) | Implements a Vector Autoregression (VAR) model for multivariate time series forecasting. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`TimeSeriesModelBase<T>`](/docs/reference/wiki/timeseries/timeseriesmodelbase/) | Provides a base class for all time series forecasting models in the library. |

## Options & Configuration (3)

| Type | Summary |
|:-----|:--------|
| [`ChronosOptions<T>`](/docs/reference/wiki/timeseries/chronosoptions/) | Options for Chronos foundation model. |
| [`DeepANTOptions<T>`](/docs/reference/wiki/timeseries/deepantoptions/) | Options for DeepANT model. |
| [`LSTMVAEOptions<T>`](/docs/reference/wiki/timeseries/lstmvaeoptions/) | Options for LSTM-VAE model. |

