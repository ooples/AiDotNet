using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Kronos (Foundation Model for the Language of Financial Markets).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Kronos is a decoder-only foundation model pre-trained on 12B+ K-line (candlestick) records
/// across 45 global exchanges. It natively understands OHLCV (Open, High, Low, Close, Volume)
/// candlestick patterns for financial market forecasting.
/// </para>
/// <para>
/// <b>Reference:</b> "Kronos: A Foundation Model for the Language of Financial Markets", 2025.
/// https://arxiv.org/abs/2508.02739
/// </para>
/// </remarks>
public class KronosOptions<T> : TimeSeriesRegressionOptions<T>
{
    public KronosOptions() { }

    public KronosOptions(KronosOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        IntermediateSize = other.IntermediateSize;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        NumCandlestickFeatures = other.NumCandlestickFeatures;
    }

    public int ContextLength { get; set; } = 1024;
    public int ForecastHorizon { get; set; } = 96;
    public int PatchLength { get; set; } = 32;
    public int HiddenDimension { get; set; } = 768;
    public int NumLayers { get; set; } = 12;
    public int NumHeads { get; set; } = 12;
    public int IntermediateSize { get; set; } = 3072;
    public double DropoutRate { get; set; } = 0.1;
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;

    /// <summary>
    /// Gets or sets the number of candlestick features (OHLCV = 5).
    /// </summary>
    /// <value>Defaults to 5 (Open, High, Low, Close, Volume).</value>
    public int NumCandlestickFeatures { get; set; } = 5;
}
