using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TS2Vec (Contrastive Learning of Universal Time Series Representations).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TS2Vec learns universal time series representations via hierarchical contrastive learning
/// across augmented context views, producing contextual representations at arbitrary granularities.
/// </para>
/// <para>
/// <b>Reference:</b> Yue et al., "TS2Vec: Towards Universal Representation of Time Series", AAAI 2022.
/// </para>
/// </remarks>
public class TS2VecOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>Creates options with the default values.</summary>
    public TS2VecOptions() { }

    /// <summary>Creates a copy of another options instance.</summary>
    /// <param name="other">The options to copy.</param>
    public TS2VecOptions(TS2VecOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        ContextLength = other.ContextLength; ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension; OutputDimension = other.OutputDimension;
        NumLayers = other.NumLayers; DropoutRate = other.DropoutRate;
        TemporalContrastiveWeight = other.TemporalContrastiveWeight;
        InstanceContrastiveWeight = other.InstanceContrastiveWeight;
    }

    /// <summary>Number of past time steps the model reads as context.</summary>
    public int ContextLength { get; set; } = 200;
    /// <summary>Number of future time steps to forecast.</summary>
    public int ForecastHorizon { get; set; } = 96;
    /// <summary>Width of the encoder's hidden representations.</summary>
    public int HiddenDimension { get; set; } = 64;
    /// <summary>Size of each time step's output representation.</summary>
    public int OutputDimension { get; set; } = 320;
    /// <summary>Number of dilated convolution blocks of the encoder.</summary>
    public int NumLayers { get; set; } = 10;
    /// <summary>Dropout probability applied during training.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Weight of the temporal contrastive loss.</summary>
    public double TemporalContrastiveWeight { get; set; } = 0.5;
    /// <summary>Weight of the instance-wise contrastive loss.</summary>
    public double InstanceContrastiveWeight { get; set; } = 0.5;
}
