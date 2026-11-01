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
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public TimeBridgeOptions() { }

    /// <summary>Creates a copy of another options instance.</summary>
    /// <param name="other">The options to copy.</param>
    public TimeBridgeOptions(TimeBridgeOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
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
        LearningRate = other.LearningRate;
    }

    /// <summary>Number of past time steps the model reads as context.</summary>
    public int ContextLength { get; set; } = 512;
    /// <summary>Number of future time steps to forecast.</summary>
    public int ForecastHorizon { get; set; } = 96;
    /// <summary>Number of time steps in each patch the series is split into.</summary>
    public int PatchLength { get; set; } = 16;
    /// <summary>Width of the model's hidden representations.</summary>
    public int HiddenDimension { get; set; } = 512;
    /// <summary>Number of transformer layers.</summary>
    public int NumLayers { get; set; } = 6;
    /// <summary>Number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;
    /// <summary>Width of the feed-forward layer inside each transformer block.</summary>
    public int IntermediateSize { get; set; } = 2048;
    /// <summary>Dropout probability applied during training.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Preset size of the foundation model.</summary>
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

    /// <summary>Gets or sets the Adam learning rate used for native training.</summary>
    /// <value>Defaults to 0.0001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns. Lower it if
    /// the loss rises instead of falling.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-4;
}
