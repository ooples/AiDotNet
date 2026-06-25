---
title: "TimeSeriesFeatureOptions"
description: "Configuration options for time series feature extraction transformers."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for time series feature extraction transformers.

## For Beginners

This class is like a settings panel for feature extraction.
You can configure:

- What statistics to calculate (mean, std, etc.)
- How large the rolling windows should be
- Whether to auto-detect optimal settings
- What lag and lead features to create

## How It Works

This unified options class configures all time series feature extractors including
rolling statistics, volatility measures, correlation calculations, and lag/lead features.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdxPeriod` | Gets or sets the ADX (Average Directional Index) period. |
| `AnnualizationFactor` | Gets or sets the annualization factor for volatility calculations. |
| `AutoDetectWindowSizes` | Gets or sets whether to automatically detect optimal window sizes. |
| `AutoDetectionMethod` | Gets or sets the auto-detection method for window sizes. |
| `BollingerBandStdDev` | Gets or sets the number of standard deviations for Bollinger Bands. |
| `CalculateMomentum` | Gets or sets whether to calculate momentum indicators. |
| `CalculateReturns` | Gets or sets whether to calculate returns (simple and log). |
| `CciPeriod` | Gets or sets the CCI (Commodity Channel Index) period. |
| `CorrelationWindowSizes` | Gets or sets the window sizes specifically for correlation calculations. |
| `CustomPercentiles` | Gets or sets custom percentiles to calculate (in addition to standard quartiles). |
| `CusumH` | Gets or sets the CUSUM decision threshold (h). |
| `CusumK` | Gets or sets the CUSUM sensitivity parameter (k). |
| `DetrendingPolynomialDegree` | Gets or sets the polynomial degree for detrending. |
| `DifferencingOrder` | Gets or sets the differencing order (number of times to difference). |
| `EdgeHandling` | Gets or sets how edge cases (beginning of series) should be handled. |
| `EnableAnomalyDetection` | Gets or sets whether to calculate anomaly detection features. |
| `EnableCorrelation` | Gets or sets whether to calculate rolling correlations. |
| `EnableDifferencing` | Gets or sets whether to apply differencing transformations. |
| `EnableRollingRegression` | Gets or sets whether to calculate rolling regression features. |
| `EnableSeasonality` | Gets or sets whether to generate seasonality and calendar features. |
| `EnableTechnicalIndicators` | Gets or sets whether to calculate technical indicators. |
| `EnableVolatility` | Gets or sets whether to calculate rolling volatility measures. |
| `EnabledAnomalyFeatures` | Gets or sets which anomaly detection features to calculate. |
| `EnabledDifferencingFeatures` | Gets or sets which differencing features to compute. |
| `EnabledIndicators` | Gets or sets which technical indicators to calculate. |
| `EnabledRegressionFeatures` | Gets or sets which rolling regression features to calculate. |
| `EnabledSeasonalityFeatures` | Gets or sets which seasonality features to generate. |
| `EnabledStatistics` | Gets or sets which rolling statistics to calculate. |
| `EnabledVolatilityMeasures` | Gets or sets which volatility measures to calculate. |
| `EwmaDecayFactor` | Gets or sets the decay factor (lambda) for EWMA volatility calculation. |
| `FeatureNameSeparator` | Gets or sets the separator for feature name components. |
| `FourierTerms` | Gets or sets the number of Fourier terms per seasonal period. |
| `FullCorrelationMatrix` | Gets or sets whether to calculate full correlation matrix or just upper triangle. |
| `GarchAlpha` | Gets or sets the alpha parameter for GARCH(1,1) model. |
| `GarchBeta` | Gets or sets the beta parameter for GARCH(1,1) model. |
| `GarchOmega` | Gets or sets the omega (constant) parameter for GARCH(1,1) model. |
| `GenerateFeatureNames` | Gets or sets whether to generate descriptive feature names. |
| `HodrickPrescottLambda` | Gets or sets the smoothing parameter (lambda) for Hodrick-Prescott filter. |
| `HolidayDates` | Gets or sets custom holiday dates to generate holiday features. |
| `HolidayWindowDays` | Gets or sets the number of days before/after holidays to flag as "near holiday". |
| `InputFeatureNames` | Gets or sets the input feature names (column names). |
| `IqrMultiplier` | Gets or sets the IQR multiplier for outlier detection. |
| `IsTradingDayData` | Gets or sets whether data represents trading days (skips weekends/holidays). |
| `IsolationForestSubsampleSize` | Gets or sets the subsample size for isolation forest. |
| `IsolationForestTrees` | Gets or sets the number of trees for isolation forest scoring. |
| `LagSteps` | Gets or sets the lag steps for lagged feature generation. |
| `LeadSteps` | Gets or sets the lead steps for leading feature generation. |
| `LongPeriod` | Gets or sets the long period for EMA/MACD calculations. |
| `MaxAutoDetectedWindows` | Gets or sets the maximum number of auto-detected window sizes. |
| `MaxWindowSize` | Gets or sets the maximum window size for auto-detection. |
| `MinWindowSize` | Gets or sets the minimum window size for auto-detection. |
| `MinimumAcceptableReturn` | Gets or sets the minimum acceptable return (MAR) for Sortino ratio calculation. |
| `OhlcColumns` | Gets or sets the OHLC column configuration for proper volatility calculations. |
| `ParallelThreshold` | Gets or sets the minimum data length to trigger parallel processing. |
| `RiskFreeRate` | Gets or sets the risk-free rate for Sharpe and Sortino ratio calculations. |
| `RiskFreeRateIsPeriodAdjusted` | Gets or sets whether the risk-free rate is already period-adjusted. |
| `RsiPeriod` | Gets or sets the RSI period. |
| `SeasonalDifferencingPeriod` | Gets or sets the seasonal period for seasonal differencing. |
| `SeasonalPeriods` | Gets or sets the seasonal periods for Fourier features. |
| `ShortPeriod` | Gets or sets the short period for EMA/MACD calculations. |
| `SignalPeriod` | Gets or sets the signal period for MACD signal line. |
| `StlRobustIterations` | Gets or sets the number of iterations for STL decomposition robustness. |
| `StlSeasonalPeriod` | Gets or sets the seasonal period for STL decomposition. |
| `StochasticDPeriod` | Gets or sets the stochastic oscillator D period (smoothing). |
| `StochasticKPeriod` | Gets or sets the stochastic oscillator K period. |
| `TimeSeriesInterval` | Gets or sets the time interval between data points for calendar feature calculation. |
| `TimeSeriesStartDate` | Gets or sets the start date of the time series for calendar calculations. |
| `UseParallelProcessing` | Gets or sets whether to use parallel processing for large datasets. |
| `WindowSizes` | Gets or sets the window sizes for rolling calculations. |
| `ZScoreThreshold` | Gets or sets the Z-score threshold for flagging anomalies. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateForFinance` | Creates a new options instance with default settings optimized for financial data. |
| `CreateMinimal` | Creates a new options instance with minimal settings for fast processing. |
| `Validate` | Validates the options and returns any validation errors. |

