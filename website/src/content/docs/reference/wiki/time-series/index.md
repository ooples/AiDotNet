---
title: "Time-Series Models"
description: "Every Time-Series Models type in AiDotNet, auto-generated with compile-checked examples."
section: "Reference"
---

Every Time-Series Models type in AiDotNet — each with a beginner-friendly explanation and, where the snippet compiles against the live library, a runnable example.

| Type | Summary |
|:-----|:--------|
| [`ARIMAModel`](/docs/reference/wiki/time-series/arimamodel/) | Implements an ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting. |
| [`ARIMAXModel`](/docs/reference/wiki/time-series/arimaxmodel/) | Implements an ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables) model for time series forecasting. |
| [`ARMAModel`](/docs/reference/wiki/time-series/armamodel/) | Implements an ARMA (AutoRegressive Moving Average) model for time series forecasting. |
| [`ARModel`](/docs/reference/wiki/time-series/armodel/) | Implements an AR (AutoRegressive) model for time series forecasting. |
| [`AutoformerCache`](/docs/reference/wiki/time-series/autoformercache/) | Cache for Autoformer forward pass computations. |
| [`AutoformerDecoderLayer`](/docs/reference/wiki/time-series/autoformerdecoderlayer/) | Autoformer decoder layer with cross-attention and series decomposition. |
| [`AutoformerEncoderLayer`](/docs/reference/wiki/time-series/autoformerencoderlayer/) | Autoformer encoder layer with series decomposition and auto-correlation. |
| [`AutoformerModel`](/docs/reference/wiki/time-series/autoformermodel/) | Implements the Autoformer model for long-term time series forecasting with decomposition. |
| [`BayesianStructuralTimeSeriesModel`](/docs/reference/wiki/time-series/bayesianstructuraltimeseriesmodel/) | Implements a Bayesian Structural Time Series model for flexible time series forecasting. |
| [`ChronosFoundationModel`](/docs/reference/wiki/time-series/chronosfoundationmodel/) | Implements the Chronos foundation model for zero-shot time series forecasting. |
| [`ChronosOptions`](/docs/reference/wiki/time-series/chronosoptions/) | Options for Chronos foundation model. |
| [`DeepARLstmCellTensor`](/docs/reference/wiki/time-series/deeparlstmcelltensor/) | Production-ready LSTM cell with proper gates (input, forget, output, cell). |
| [`DeepARModel`](/docs/reference/wiki/time-series/deeparmodel/) | Implements DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks. |
| [`DistillingConvTensor`](/docs/reference/wiki/time-series/distillingconvtensor/) | Tensor-based distilling convolution layer for sequence compression. |
| [`DLinearModel`](/docs/reference/wiki/time-series/dlinearmodel/) | DLinear — decomposition-linear forecaster (Zeng et al., AAAI 2023, "Are Transformers Effective for Time Series Forecasting?"). |
| [`DynamicRegressionWithARIMAErrors`](/docs/reference/wiki/time-series/dynamicregressionwitharimaerrors/) | Implements a Dynamic Regression model with ARIMA errors for time series forecasting. |
| [`ExponentialSmoothingModel`](/docs/reference/wiki/time-series/exponentialsmoothingmodel/) | Represents a model that implements exponential smoothing for time series forecasting. |
| [`GARCHModel`](/docs/reference/wiki/time-series/garchmodel/) | Represents a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model for time series with changing volatility. |
| [`InformerDecoderLayerTensor`](/docs/reference/wiki/time-series/informerdecoderlayertensor/) | Tensor-based decoder layer for Informer with cross-attention. |
| [`InformerEncoderLayerTensor`](/docs/reference/wiki/time-series/informerencoderlayertensor/) | Tensor-based encoder layer for Informer with ProbSparse attention. |
| [`InformerModel`](/docs/reference/wiki/time-series/informermodel/) | Implements the Informer model for efficient long-sequence time series forecasting. |
| [`InterventionAnalysisModel`](/docs/reference/wiki/time-series/interventionanalysismodel/) | Represents a model that analyzes and forecasts time series data with interventions or structural changes. |
| [`MAModel`](/docs/reference/wiki/time-series/mamodel/) | Implements a Moving Average (MA) model for time series forecasting. |
| [`NBEATSBlock`](/docs/reference/wiki/time-series/nbeatsblock/) | Represents a single block in the N-BEATS architecture. |
| [`NBEATSModel`](/docs/reference/wiki/time-series/nbeatsmodel/) | Implements the N-BEATS (Neural Basis Expansion Analysis for Time Series) model for forecasting. |
| [`NeuralNetworkARIMAModel`](/docs/reference/wiki/time-series/neuralnetworkarimamodel/) | Represents a Neural Network ARIMA (Autoregressive Integrated Moving Average) model for time series forecasting. |
| [`NHiTSModel`](/docs/reference/wiki/time-series/nhitsmodel/) | Implements N-HiTS (Neural Hierarchical Interpolation for Time Series) for efficient long-horizon forecasting. |
| [`NHiTSStackTensor`](/docs/reference/wiki/time-series/nhitsstacktensor/) | Represents a single stack in the N-HiTS architecture using Tensor operations. |
| [`NLinearModel`](/docs/reference/wiki/time-series/nlinearmodel/) | NLinear — normalization-linear forecaster (Zeng et al., AAAI 2023). |
| [`ProphetModel`](/docs/reference/wiki/time-series/prophetmodel/) | Represents a Prophet model for time series forecasting. |
| [`SARIMAModel`](/docs/reference/wiki/time-series/sarimamodel/) | Implements a Seasonal Autoregressive Integrated Moving Average (SARIMA) model for time series forecasting. |
| [`SpectralAnalysisModel`](/docs/reference/wiki/time-series/spectralanalysismodel/) | Implements spectral analysis for time series data, which transforms time domain signals into the frequency domain. |
| [`StateSpaceModel`](/docs/reference/wiki/time-series/statespacemodel/) | Implements a State Space Model for time series analysis and forecasting. |
| [`STLDecomposition`](/docs/reference/wiki/time-series/stldecomposition/) | Implements Seasonal-Trend decomposition using LOESS (STL) for time series analysis. |
| [`TBATSModel`](/docs/reference/wiki/time-series/tbatsmodel/) | Implements the TBATS (Trigonometric, Box-Cox transform, ARMA errors, Trend, and Seasonal components) model for complex time series forecasting with multiple seasonal patterns. |
| [`TemporalFusionTransformer`](/docs/reference/wiki/time-series/temporalfusiontransformer/) | Implements the Temporal Fusion Transformer (TFT) per Lim et al. |
| [`TiDEModel`](/docs/reference/wiki/time-series/tidemodel/) | TiDE — Time-series Dense Encoder (Das et al., TMLR 2023). |
| [`TimeSeriesModelBase`](/docs/reference/wiki/time-series/timeseriesmodelbase/) | Provides a base class for all time series forecasting models in the library. |
| [`TransferFunctionModel`](/docs/reference/wiki/time-series/transferfunctionmodel/) | Implements a Transfer Function Model for time series analysis, which combines ARIMA modeling with external input variables to capture dynamic relationships between multiple time series. |
| [`TransformerGraphHelper`](/docs/reference/wiki/time-series/transformergraphhelper/) | Shared computation graph builders for transformer layer components. |
| [`UnobservedComponentsModel`](/docs/reference/wiki/time-series/unobservedcomponentsmodel/) | Implements an Unobserved Components Model (UCM) for time series decomposition and forecasting. |
| [`VARMAModel`](/docs/reference/wiki/time-series/varmamodel/) | Implements a Vector Autoregressive Moving Average (VARMA) model for multivariate time series forecasting. |
| [`VectorAutoRegressionModel`](/docs/reference/wiki/time-series/vectorautoregressionmodel/) | Implements a Vector Autoregression (VAR) model for multivariate time series forecasting. |
