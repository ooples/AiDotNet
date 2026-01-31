using System;
using System.Collections.Generic;

namespace AiDotNet.Finance.Data;

/// <summary>
/// Factory helpers for creating financial data loaders.
/// </summary>
/// <remarks>
/// <para>
/// This factory keeps common loader configurations in one place so callers can
/// create forecasting loaders without repeating boilerplate setup.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use these helpers when you have OHLCV price data and want
/// a ready-to-use FinancialDataLoader for training or evaluation.
/// </para>
/// </remarks>
public static class FinancialDataLoaderFactory
{
    /// <summary>
    /// Creates a financial data loader from a list of market data points.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="series">The OHLCV series to load.</param>
    /// <param name="sequenceLength">The lookback window size.</param>
    /// <param name="predictionHorizon">The number of steps to predict.</param>
    /// <param name="includeVolume">Whether to include volume as a feature.</param>
    /// <param name="includeReturns">Whether to include returns as a feature.</param>
    /// <param name="predictReturns">Whether to predict returns instead of prices.</param>
    /// <param name="normalizeMinMax">Whether to apply min-max normalization to features.</param>
    /// <param name="preprocessor">Optional preprocessor instance.</param>
    /// <param name="batchSize">Batch size for iteration.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the fastest way to go from a list of OHLCV bars
    /// to a loader that yields input/output tensors for forecasting models.
    /// </para>
    /// </remarks>
    public static FinancialDataLoader<T> FromSeries<T>(
        IReadOnlyList<MarketDataPoint<T>> series,
        int sequenceLength,
        int predictionHorizon,
        bool includeVolume = true,
        bool includeReturns = false,
        bool predictReturns = false,
        bool normalizeMinMax = false,
        FinancialPreprocessor<T>? preprocessor = null,
        int batchSize = 32)
    {
        if (series is null)
        {
            throw new ArgumentNullException(nameof(series));
        }

        return new FinancialDataLoader<T>(
            series,
            sequenceLength,
            predictionHorizon,
            includeVolume,
            includeReturns,
            predictReturns,
            normalizeMinMax,
            preprocessor,
            batchSize);
    }

    /// <summary>
    /// Creates a financial data loader from a market data provider.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="provider">Provider containing OHLCV data.</param>
    /// <param name="sequenceLength">The lookback window size.</param>
    /// <param name="predictionHorizon">The number of steps to predict.</param>
    /// <param name="includeVolume">Whether to include volume as a feature.</param>
    /// <param name="includeReturns">Whether to include returns as a feature.</param>
    /// <param name="predictReturns">Whether to predict returns instead of prices.</param>
    /// <param name="normalizeMinMax">Whether to apply min-max normalization to features.</param>
    /// <param name="preprocessor">Optional preprocessor instance.</param>
    /// <param name="batchSize">Batch size for iteration.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If your data is already stored in a MarketDataProvider,
    /// this helper extracts the series and builds the loader for you.
    /// </para>
    /// </remarks>
    public static FinancialDataLoader<T> FromProvider<T>(
        MarketDataProvider<T> provider,
        int sequenceLength,
        int predictionHorizon,
        bool includeVolume = true,
        bool includeReturns = false,
        bool predictReturns = false,
        bool normalizeMinMax = false,
        FinancialPreprocessor<T>? preprocessor = null,
        int batchSize = 32)
    {
        if (provider is null)
        {
            throw new ArgumentNullException(nameof(provider));
        }

        return FromSeries(
            provider.GetAll(),
            sequenceLength,
            predictionHorizon,
            includeVolume,
            includeReturns,
            predictReturns,
            normalizeMinMax,
            preprocessor,
            batchSize);
    }
}
