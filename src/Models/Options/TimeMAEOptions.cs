using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TimeMAE (Masked Autoencoder for Time Series).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TimeMAE applies masked autoencoding to time series, randomly masking patches of the input
/// and training a transformer to reconstruct the missing patches, learning rich temporal representations.
/// </para>
/// <para>
/// <b>Reference:</b> Cheng et al., "TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders", 2023.
/// </para>
/// </remarks>
public class TimeMAEOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>Creates options with the default values.</summary>
    public TimeMAEOptions() { }

    /// <summary>Creates a copy of another options instance.</summary>
    /// <param name="other">The options to copy.</param>
    public TimeMAEOptions(TimeMAEOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        ContextLength = other.ContextLength; ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength; HiddenDimension = other.HiddenDimension;
        NumEncoderLayers = other.NumEncoderLayers; NumDecoderLayers = other.NumDecoderLayers;
        NumHeads = other.NumHeads; MaskRatio = other.MaskRatio;
        DropoutRate = other.DropoutRate; LearningRate = other.LearningRate;
    }

    /// <summary>Number of past time steps the model reads as context.</summary>
    public int ContextLength { get; set; } = 512;
    /// <summary>Number of future time steps to forecast.</summary>
    public int ForecastHorizon { get; set; } = 96;
    /// <summary>Number of time steps in each patch the series is split into.</summary>
    public int PatchLength { get; set; } = 16;
    /// <summary>Width of the model's hidden representations.</summary>
    public int HiddenDimension { get; set; } = 256;
    /// <summary>Number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 6;
    /// <summary>Number of decoder layers.</summary>
    public int NumDecoderLayers { get; set; } = 2;
    /// <summary>Number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;
    /// <summary>Fraction of patches masked in each training series.</summary>
    public double MaskRatio { get; set; } = 0.75;
    /// <summary>Dropout probability applied during training.</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the Adam learning rate used for native training.</summary>
    /// <value>Defaults to 0.0001.</value>
    public double LearningRate { get; set; } = 1e-4;
}
