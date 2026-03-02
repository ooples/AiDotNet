using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TimeBridge (Non-Stationarity Matters for Time Series Foundation Models).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TimeBridge addresses the critical non-stationarity gap in time series foundation models.
/// It introduces a bridge mechanism that preserves and restores non-stationary information
/// (trends, level shifts) that is typically lost during standard normalization.
/// </para>
/// <para>
/// <b>Reference:</b> "TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting", 2024.
/// </para>
/// </remarks>
public class TimeBridgeOptions<T> : TimeSeriesRegressionOptions<T>
{
    public TimeBridgeOptions() { }

    public TimeBridgeOptions(TimeBridgeOptions<T> other)
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
        BridgeDimension = other.BridgeDimension;
        UseStationarityGating = other.UseStationarityGating;
    }

    public int ContextLength { get; set; } = 512;
    public int ForecastHorizon { get; set; } = 96;
    public int PatchLength { get; set; } = 16;
    public int HiddenDimension { get; set; } = 512;
    public int NumLayers { get; set; } = 6;
    public int NumHeads { get; set; } = 8;
    public int IntermediateSize { get; set; } = 2048;
    public double DropoutRate { get; set; } = 0.1;
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;

    /// <summary>
    /// Gets or sets the dimension of the non-stationarity bridge module.
    /// </summary>
    /// <value>Defaults to 128.</value>
    public int BridgeDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets whether to use stationarity gating for adaptive restoration.
    /// </summary>
    /// <value>Defaults to true.</value>
    public bool UseStationarityGating { get; set; } = true;
}
