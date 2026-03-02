using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for MG-TSD (Multi-Granularity Time Series Diffusion).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MG-TSD introduces a multi-granularity guidance diffusion model that captures temporal
/// patterns at different scales. It uses a coarse-to-fine guidance mechanism where
/// predictions at coarser granularities guide the fine-grained diffusion process.
/// </para>
/// <para>
/// <b>Reference:</b> Fan et al., "MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process", ICLR 2024.
/// </para>
/// </remarks>
public class MGTSDOptions<T> : TimeSeriesRegressionOptions<T>
{
    public MGTSDOptions() { }

    public MGTSDOptions(MGTSDOptions<T> other)
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
        NumGranularities = other.NumGranularities;
        GuidanceWeight = other.GuidanceWeight;
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
    /// Gets or sets the number of temporal granularity levels for guidance.
    /// </summary>
    /// <value>Defaults to 3 (e.g., hourly, daily, weekly).</value>
    public int NumGranularities { get; set; } = 3;

    /// <summary>
    /// Gets or sets the weight for cross-granularity guidance.
    /// </summary>
    /// <value>Defaults to 0.5.</value>
    public double GuidanceWeight { get; set; } = 0.5;
}
