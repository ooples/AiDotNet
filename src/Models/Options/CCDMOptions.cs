using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for CCDM (Conditional Continuous Diffusion Model for Time Series).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CCDM extends continuous diffusion models for conditional time series generation.
/// It operates in continuous space (unlike discrete token-based approaches) and uses
/// a score-matching objective for high-quality probabilistic forecasting.
/// </para>
/// </remarks>
public class CCDMOptions<T> : TimeSeriesRegressionOptions<T>
{
    public CCDMOptions() { }

    public CCDMOptions(CCDMOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        DiffusionSteps = other.DiffusionSteps;
        DropoutRate = other.DropoutRate;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        SigmaMin = other.SigmaMin;
        SigmaMax = other.SigmaMax;
    }

    public int ContextLength { get; set; } = 168;
    public int ForecastHorizon { get; set; } = 24;
    public int HiddenDimension { get; set; } = 128;
    public int NumLayers { get; set; } = 4;
    public int NumHeads { get; set; } = 8;
    public int DiffusionSteps { get; set; } = 100;
    public double DropoutRate { get; set; } = 0.1;
    public double BetaStart { get; set; } = 0.0001;
    public double BetaEnd { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the minimum noise level for the continuous diffusion schedule.
    /// </summary>
    /// <value>Defaults to 0.002.</value>
    public double SigmaMin { get; set; } = 0.002;

    /// <summary>
    /// Gets or sets the maximum noise level for the continuous diffusion schedule.
    /// </summary>
    /// <value>Defaults to 80.0.</value>
    public double SigmaMax { get; set; } = 80.0;
}
