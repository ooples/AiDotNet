using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;

namespace AiDotNet.Finance.Data;

/// <summary>
/// Represents a single market data point (OHLCV).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class stores one time-stamped bar of market data, including open, high,
/// low, close, and volume values. It is the basic unit of financial time series
/// used by data loaders and trading environments.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as one row in a stock chart: it tells you
/// the price when the period opened, the highest and lowest prices, the closing
/// price, and how much volume traded during that period.
/// </para>
/// </remarks>
public sealed class MarketDataPoint<T>
{
    /// <summary>
    /// Gets the timestamp for the market data point.
    /// </summary>
    public DateTime Timestamp { get; }

    /// <summary>
    /// Gets the open price.
    /// </summary>
    public T Open { get; }

    /// <summary>
    /// Gets the high price.
    /// </summary>
    public T High { get; }

    /// <summary>
    /// Gets the low price.
    /// </summary>
    public T Low { get; }

    /// <summary>
    /// Gets the close price.
    /// </summary>
    public T Close { get; }

    /// <summary>
    /// Gets the traded volume.
    /// </summary>
    public T Volume { get; }

    /// <summary>
    /// Creates a new market data point.
    /// </summary>
    /// <param name="timestamp">The timestamp for the data point.</param>
    /// <param name="open">The open price.</param>
    /// <param name="high">The high price.</param>
    /// <param name="low">The low price.</param>
    /// <param name="close">The close price.</param>
    /// <param name="volume">The traded volume.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor packages all the prices and volume
    /// for a single time step into one object so later steps (like preprocessing)
    /// can read them easily.
    /// </para>
    /// </remarks>
    public MarketDataPoint(DateTime timestamp, T open, T high, T low, T close, T volume)
    {
        Timestamp = timestamp;
        Open = open;
        High = high;
        Low = low;
        Close = close;
        Volume = volume;
    }
}

/// <summary>
/// Stores and serves market data for finance workflows.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MarketDataProvider is a lightweight in-memory store for OHLCV data.
/// It is intentionally simple so it can be used by loaders, preprocessors,
/// and trading environments without extra dependencies.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is like a small database of price bars. You can
/// add data points, slice out date ranges, or convert the series into tensors
/// for model training.
/// </para>
/// </remarks>
public sealed class MarketDataProvider<T>
{
    private readonly List<MarketDataPoint<T>> _points = new();

    /// <summary>
    /// Gets the number of data points stored.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many time steps of data are loaded.
    /// </para>
    /// </remarks>
    public int Count => _points.Count;

    /// <summary>
    /// Adds a single market data point.
    /// </summary>
    /// <param name="point">The market data point to add.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this to append a new price bar to the data store.
    /// </para>
    /// </remarks>
    public void Add(MarketDataPoint<T> point)
    {
        if (point == null)
        {
            throw new ArgumentNullException(nameof(point));
        }

        _points.Add(point);
    }

    /// <summary>
    /// Adds multiple market data points at once.
    /// </summary>
    /// <param name="points">The collection of points to add.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a faster way to load a full historical series
    /// instead of adding points one by one.
    /// </para>
    /// </remarks>
    public void AddRange(IEnumerable<MarketDataPoint<T>> points)
    {
        if (points == null)
        {
            throw new ArgumentNullException(nameof(points));
        }

        _points.AddRange(points);
    }

    /// <summary>
    /// Clears all stored market data points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This resets the provider back to an empty state.
    /// </para>
    /// </remarks>
    public void Clear()
    {
        _points.Clear();
    }

    /// <summary>
    /// Returns all stored points as a read-only list.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you want to iterate through all
    /// available market data.
    /// </para>
    /// </remarks>
    public IReadOnlyList<MarketDataPoint<T>> GetAll()
    {
        return _points;
    }

    /// <summary>
    /// Returns points within a specific time range.
    /// </summary>
    /// <param name="start">Inclusive start timestamp.</param>
    /// <param name="end">Inclusive end timestamp.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This lets you slice the time series to just the period
    /// you care about, such as the last year or a specific market regime.
    /// </para>
    /// </remarks>
    public IReadOnlyList<MarketDataPoint<T>> GetRange(DateTime start, DateTime end)
    {
        return _points
            .Where(point => point.Timestamp >= start && point.Timestamp <= end)
            .ToList();
    }

    /// <summary>
    /// Returns a fixed-size window of points by index.
    /// </summary>
    /// <param name="startIndex">The index of the first point in the window.</param>
    /// <param name="length">How many points to return.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is useful for making rolling windows of data
    /// (e.g., the last 30 days of prices).
    /// </para>
    /// </remarks>
    public IReadOnlyList<MarketDataPoint<T>> GetWindow(int startIndex, int length)
    {
        if (startIndex < 0 || startIndex >= _points.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(startIndex));
        }

        if (length < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(length));
        }

        int safeLength = Math.Min(length, _points.Count - startIndex);
        return _points.Skip(startIndex).Take(safeLength).ToList();
    }

    /// <summary>
    /// Converts the stored OHLCV data into a tensor.
    /// </summary>
    /// <param name="includeVolume">Whether to include volume as a feature.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Many models expect numeric tensors. This method
    /// converts the market data into a matrix-like tensor where each row is
    /// a time step and each column is a feature (open, high, low, close, volume).
    /// </para>
    /// </remarks>
    public Tensor<T> ToTensor(bool includeVolume = true)
    {
        int featureCount = GetFeatureCount(includeVolume);
        var data = new T[_points.Count * featureCount];

        for (int i = 0; i < _points.Count; i++)
        {
            int offset = i * featureCount;
            var point = _points[i];

            data[offset] = point.Open;
            data[offset + 1] = point.High;
            data[offset + 2] = point.Low;
            data[offset + 3] = point.Close;

            if (includeVolume)
            {
                data[offset + 4] = point.Volume;
            }
        }

        return new Tensor<T>(new[] { _points.Count, featureCount }, new Vector<T>(data));
    }

    /// <summary>
    /// Computes the feature count for OHLCV tensors.
    /// </summary>
    /// <param name="includeVolume">Whether volume is included.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper keeps the feature count consistent when
    /// you include or exclude volume.
    /// </para>
    /// </remarks>
    private static int GetFeatureCount(bool includeVolume)
    {
        return includeVolume ? 5 : 4;
    }
}
