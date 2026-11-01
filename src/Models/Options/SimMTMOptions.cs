using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for SimMTM (Simple Pre-Training Framework for Masked Time-Series Modeling).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SimMTM combines masked time series modeling with series-level similarity learning,
/// recovering masked series by aggregating from similar unmasked series in the batch.
/// </para>
/// <para>
/// <b>Reference:</b> Dong et al., "SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling", NeurIPS 2023.
/// </para>
/// </remarks>
public class SimMTMOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>Creates options with the default values.</summary>
    public SimMTMOptions() { }

    /// <summary>Creates a copy of another options instance.</summary>
    /// <param name="other">The options to copy.</param>
    public SimMTMOptions(SimMTMOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        ContextLength = other.ContextLength; ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength; HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers; NumHeads = other.NumHeads;
        MaskRatio = other.MaskRatio; DropoutRate = other.DropoutRate;
        SimilarityTemperature = other.SimilarityTemperature;
    }

    /// <summary>Number of past time steps the model reads as context.</summary>
    public int ContextLength { get; set; } = 512;
    /// <summary>Number of future time steps to forecast.</summary>
    public int ForecastHorizon { get; set; } = 96;
    /// <summary>Number of time steps in each patch the series is split into.</summary>
    public int PatchLength { get; set; } = 16;
    /// <summary>Width of the encoder's hidden representations.</summary>
    public int HiddenDimension { get; set; } = 256;
    /// <summary>Number of encoder layers.</summary>
    public int NumLayers { get; set; } = 6;
    /// <summary>Number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;
    /// <summary>Fraction of time steps masked in each training series.</summary>
    public double MaskRatio { get; set; } = 0.5;
    /// <summary>Dropout probability applied during training.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Temperature of the series-wise similarity used to weight neighbors when reconstructing masked series.</summary>
    public double SimilarityTemperature { get; set; } = 0.07;
}
