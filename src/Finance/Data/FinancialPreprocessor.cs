using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;

namespace AiDotNet.Finance.Data;

/// <summary>
/// Preprocesses financial time series data into model-ready tensors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FinancialPreprocessor converts raw OHLCV data into feature tensors,
/// builds rolling windows for supervised learning, and provides basic
/// normalization utilities for time series.
/// </para>
/// <para>
/// <b>For Beginners:</b> This class is the "data prep" step. It turns
/// messy price bars into clean numeric tensors that models can learn from.
/// It also supports windowing (lookback) and scaling (normalization).
/// </para>
/// </remarks>
public sealed class FinancialPreprocessor<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new financial preprocessor.
    /// </summary>
    /// <param name="numOps">Optional numeric operations implementation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The preprocessor needs math helpers for the numeric type
    /// (float, double, etc.). If you do not provide one, it uses the default helpers.
    /// </para>
    /// </remarks>
    public FinancialPreprocessor(INumericOperations<T>? numOps = null)
    {
        _numOps = numOps ?? MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the number of features produced for each time step.
    /// </summary>
    /// <param name="includeVolume">Whether volume is included as a feature.</param>
    /// <param name="includeReturns">Whether returns are included as a feature.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many numbers describe each time step
    /// (e.g., 4 for OHLC, 5 for OHLCV, 6 for OHLCV+returns).
    /// </para>
    /// </remarks>
    public int GetFeatureCount(bool includeVolume, bool includeReturns)
    {
        int baseCount = includeVolume ? 5 : 4;
        return includeReturns ? baseCount + 1 : baseCount;
    }

    /// <summary>
    /// Converts raw market data into a 2D feature tensor.
    /// </summary>
    /// <param name="points">The market data points to transform.</param>
    /// <param name="includeVolume">Whether to include volume as a feature.</param>
    /// <param name="includeReturns">Whether to include close-to-close returns as a feature.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This turns the list of OHLCV points into a matrix where
    /// each row is a time step and each column is a feature.
    /// </para>
    /// </remarks>
    public Tensor<T> CreateFeatureTensor(
        IReadOnlyList<MarketDataPoint<T>> points,
        bool includeVolume = true,
        bool includeReturns = false)
    {
        ValidateSeries(points, nameof(points));

        int featureCount = GetFeatureCount(includeVolume, includeReturns);
        var data = new T[points.Count * featureCount];

        for (int i = 0; i < points.Count; i++)
        {
            var features = BuildFeatureVector(points, i, includeVolume, includeReturns);
            int offset = i * featureCount;
            for (int f = 0; f < featureCount; f++)
            {
                data[offset + f] = features[f];
            }
        }

        return new Tensor<T>(new[] { points.Count, featureCount }, new Vector<T>(data));
    }

    /// <summary>
    /// Creates windowed features and targets for supervised forecasting.
    /// </summary>
    /// <param name="points">The market data points to transform.</param>
    /// <param name="sequenceLength">The lookback window length.</param>
    /// <param name="predictionHorizon">How many future steps to predict.</param>
    /// <param name="includeVolume">Whether to include volume as a feature.</param>
    /// <param name="includeReturns">Whether to include returns as a feature.</param>
    /// <param name="predictReturns">Whether the target is returns instead of close price.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main step for time series forecasting. It slices
    /// the series into input windows (past data) and target windows (future data).
    /// </para>
    /// </remarks>
    public (Tensor<T> Features, Tensor<T> Targets) CreateSupervisedLearningTensors(
        IReadOnlyList<MarketDataPoint<T>> points,
        int sequenceLength,
        int predictionHorizon,
        bool includeVolume = true,
        bool includeReturns = false,
        bool predictReturns = false)
    {
        ValidateSeries(points, nameof(points));
        ValidateWindowSizes(sequenceLength, predictionHorizon);

        int featureCount = GetFeatureCount(includeVolume, includeReturns);
        int sampleCount = points.Count - sequenceLength - predictionHorizon + 1;
        if (sampleCount < 1)
        {
            return (new Tensor<T>(new[] { 0, sequenceLength, featureCount }),
                new Tensor<T>(new[] { 0, predictionHorizon, 1 }));
        }

        var featureData = new T[sampleCount * sequenceLength * featureCount];
        var targetData = new T[sampleCount * predictionHorizon];

        for (int sample = 0; sample < sampleCount; sample++)
        {
            for (int t = 0; t < sequenceLength; t++)
            {
                var features = BuildFeatureVector(points, sample + t, includeVolume, includeReturns);
                int baseOffset = (sample * sequenceLength + t) * featureCount;
                for (int f = 0; f < featureCount; f++)
                {
                    featureData[baseOffset + f] = features[f];
                }
            }

            for (int h = 0; h < predictionHorizon; h++)
            {
                int targetIndex = sample + sequenceLength + h;
                T targetValue = predictReturns
                    ? ComputeReturn(points, targetIndex)
                    : points[targetIndex].Close;

                targetData[(sample * predictionHorizon) + h] = targetValue;
            }
        }

        var featuresTensor = new Tensor<T>(new[] { sampleCount, sequenceLength, featureCount }, new Vector<T>(featureData));
        var targetsTensor = new Tensor<T>(new[] { sampleCount, predictionHorizon, 1 }, new Vector<T>(targetData));
        return (featuresTensor, targetsTensor);
    }

    /// <summary>
    /// Applies min-max normalization across the feature dimension.
    /// </summary>
    /// <param name="input">The tensor to normalize.</param>
    /// <param name="stats">Returns the min and max per feature.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Min-max scaling squeezes each feature to the 0-1 range.
    /// This helps models train more stably when features are on different scales.
    /// </para>
    /// </remarks>
    public Tensor<T> NormalizeMinMax(Tensor<T> input, out (Vector<T> Min, Vector<T> Max) stats)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Length == 0)
        {
            stats = (new Vector<T>(0), new Vector<T>(0));
            return input;
        }

        int featureCount = input.Shape[^1];
        var min = new T[featureCount];
        var max = new T[featureCount];

        InitializeMinMax(input, min, max, featureCount);

        for (int i = 0; i < input.Length; i++)
        {
            int feature = i % featureCount;
            T value = input.Data.Span[i];
            if (_numOps.Compare(value, min[feature]) < 0)
            {
                min[feature] = value;
            }
            if (_numOps.Compare(value, max[feature]) > 0)
            {
                max[feature] = value;
            }
        }

        var normalized = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            int feature = i % featureCount;
            T range = _numOps.Subtract(max[feature], min[feature]);
            T safeRange = _numOps.Compare(range, _numOps.Zero) == 0 ? _numOps.One : range;
            normalized[i] = _numOps.Divide(_numOps.Subtract(input.Data.Span[i], min[feature]), safeRange);
        }

        stats = (new Vector<T>(min), new Vector<T>(max));
        return new Tensor<T>(input.Shape, new Vector<T>(normalized));
    }

    /// <summary>
    /// Applies z-score normalization across the feature dimension.
    /// </summary>
    /// <param name="input">The tensor to normalize.</param>
    /// <param name="stats">Returns the mean and standard deviation per feature.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Z-score scaling makes features have zero mean and unit
    /// variance, which helps models treat all features equally.
    /// </para>
    /// </remarks>
    public Tensor<T> NormalizeZScore(Tensor<T> input, out (Vector<T> Mean, Vector<T> StdDev) stats)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Length == 0)
        {
            stats = (new Vector<T>(0), new Vector<T>(0));
            return input;
        }

        int featureCount = input.Shape[^1];
        int sampleCount = input.Length / featureCount;
        var mean = new T[featureCount];
        var std = new T[featureCount];

        for (int i = 0; i < input.Length; i++)
        {
            int feature = i % featureCount;
            mean[feature] = _numOps.Add(mean[feature], input.Data.Span[i]);
        }

        for (int f = 0; f < featureCount; f++)
        {
            mean[f] = _numOps.Divide(mean[f], _numOps.FromDouble(sampleCount));
        }

        for (int i = 0; i < input.Length; i++)
        {
            int feature = i % featureCount;
            T diff = _numOps.Subtract(input.Data.Span[i], mean[feature]);
            std[feature] = _numOps.Add(std[feature], _numOps.Multiply(diff, diff));
        }

        double denom = Math.Max(1, sampleCount - 1);
        for (int f = 0; f < featureCount; f++)
        {
            std[f] = _numOps.Sqrt(_numOps.Divide(std[f], _numOps.FromDouble(denom)));
            if (_numOps.Compare(std[f], _numOps.Zero) == 0)
            {
                std[f] = _numOps.One;
            }
        }

        var normalized = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            int feature = i % featureCount;
            normalized[i] = _numOps.Divide(_numOps.Subtract(input.Data.Span[i], mean[feature]), std[feature]);
        }

        stats = (new Vector<T>(mean), new Vector<T>(std));
        return new Tensor<T>(input.Shape, new Vector<T>(normalized));
    }

    /// <summary>
    /// Builds a feature vector for a single time step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This gathers the numbers that describe one time step,
    /// such as prices, volume, and optional returns.
    /// </para>
    /// </remarks>
    private Vector<T> BuildFeatureVector(
        IReadOnlyList<MarketDataPoint<T>> points,
        int index,
        bool includeVolume,
        bool includeReturns)
    {
        int featureCount = GetFeatureCount(includeVolume, includeReturns);
        var values = new T[featureCount];
        var point = points[index];

        values[0] = point.Open;
        values[1] = point.High;
        values[2] = point.Low;
        values[3] = point.Close;

        int offset = 4;
        if (includeVolume)
        {
            values[offset] = point.Volume;
            offset++;
        }

        if (includeReturns)
        {
            values[offset] = ComputeReturn(points, index);
        }

        return new Vector<T>(values);
    }

    /// <summary>
    /// Computes the close-to-close return for a given index.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns measure how much the price changed compared to
    /// the previous time step. They are often more stable than raw prices.
    /// </para>
    /// </remarks>
    private T ComputeReturn(IReadOnlyList<MarketDataPoint<T>> points, int index)
    {
        if (index <= 0)
        {
            return _numOps.Zero;
        }

        T prev = points[index - 1].Close;
        T current = points[index].Close;
        T denom = _numOps.Compare(_numOps.Abs(prev), _numOps.FromDouble(1e-12)) > 0 ? prev : _numOps.FromDouble(1e-12);
        return _numOps.Divide(_numOps.Subtract(current, prev), denom);
    }

    /// <summary>
    /// Validates that the series is non-empty.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> We need at least one data point before we can build
    /// features or targets.
    /// </para>
    /// </remarks>
    private static void ValidateSeries(IReadOnlyList<MarketDataPoint<T>> points, string paramName)
    {
        if (points == null)
        {
            throw new ArgumentNullException(paramName);
        }

        if (points.Count == 0)
        {
            throw new ArgumentException("Market data series cannot be empty.", paramName);
        }
    }

    /// <summary>
    /// Validates the lookback and horizon settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The model needs positive window sizes to learn from
    /// the past and predict the future.
    /// </para>
    /// </remarks>
    private static void ValidateWindowSizes(int sequenceLength, int predictionHorizon)
    {
        if (sequenceLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceLength));
        }

        if (predictionHorizon <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon));
        }
    }

    /// <summary>
    /// Initializes min/max arrays from the first feature row.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> We need a starting value for each feature before
    /// we scan the data to find true minima and maxima.
    /// </para>
    /// </remarks>
    private void InitializeMinMax(Tensor<T> input, T[] min, T[] max, int featureCount)
    {
        for (int f = 0; f < featureCount; f++)
        {
            min[f] = input.Data.Span[f];
            max[f] = input.Data.Span[f];
        }
    }
}
