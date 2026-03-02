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
    public TS2VecOptions() { }

    public TS2VecOptions(TS2VecOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength; ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension; OutputDimension = other.OutputDimension;
        NumLayers = other.NumLayers; DropoutRate = other.DropoutRate;
        TemporalContrastiveWeight = other.TemporalContrastiveWeight;
        InstanceContrastiveWeight = other.InstanceContrastiveWeight;
    }

    public int ContextLength { get; set; } = 200;
    public int ForecastHorizon { get; set; } = 96;
    public int HiddenDimension { get; set; } = 64;
    public int OutputDimension { get; set; } = 320;
    public int NumLayers { get; set; } = 10;
    public double DropoutRate { get; set; } = 0.1;
    public double TemporalContrastiveWeight { get; set; } = 0.5;
    public double InstanceContrastiveWeight { get; set; } = 0.5;
}
