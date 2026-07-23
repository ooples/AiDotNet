using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Time-MoE (Billion-Scale Time Series Foundation Models with Mixture of Experts).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Time-MoE is the first billion-scale time series foundation model, using sparse Mixture
/// of Experts (MoE) for efficient scaling up to 2.4B parameters. It uses a decoder-only
/// transformer with MoE feed-forward layers.
/// </para>
/// <para><b>For Beginners:</b> Time-MoE achieves massive scale efficiently:
///
/// <b>Mixture of Experts:</b>
/// Instead of using one large feed-forward network, MoE uses multiple smaller "expert"
/// networks and a router that selects which experts to use for each input. This means
/// the model has many parameters but only uses a fraction for each prediction.
///
/// <b>Model Sizes:</b>
/// - 50M parameters (all active)
/// - 200M total / ~50M active per token
/// - 2.4B total / ~300M active per token (largest)
/// </para>
/// <para>
/// <b>Reference:</b> Shi et al., "Time-MoE: Billion-Scale Time Series Foundation Models
/// with Mixture of Experts", ICLR 2025. https://openreview.net/forum?id=e1wDDFmlVu
/// </para>
/// </remarks>
public class TimeMoEOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public TimeMoEOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public TimeMoEOptions(TimeMoEOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        IntermediateSize = other.IntermediateSize;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        NumExperts = other.NumExperts;
        NumActiveExperts = other.NumActiveExperts;
        RouterAuxLossWeight = other.RouterAuxLossWeight;
    }

    /// <summary>
    /// Gets or sets the context length.
    /// </summary>
    /// <value>Defaults to 2048.</value>
    public int ContextLength { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the forecast horizon.
    /// </summary>
    /// <value>Defaults to 96.</value>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the patch length.
    /// </summary>
    /// <value>Defaults to 32.</value>
    public int PatchLength { get; set; } = 32;

    /// <summary>
    /// Gets or sets the hidden dimension.
    /// </summary>
    /// <value>Defaults to 1024.</value>
    public int HiddenDimension { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Defaults to 24.</value>
    public int NumLayers { get; set; } = 24;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 16.</value>
    public int NumHeads { get; set; } = 16;

    /// <summary>
    /// Gets or sets the intermediate size per expert.
    /// </summary>
    /// <value>Defaults to 4096.</value>
    public int IntermediateSize { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>Defaults to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>Defaults to <see cref="FoundationModelSize.Large"/>.</value>
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Large;

    /// <summary>
    /// Gets or sets the total number of experts in each MoE layer.
    /// </summary>
    /// <value>Defaults to 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each MoE layer contains this many expert networks.
    /// More experts = more total parameters but same active parameters per token.
    /// </para>
    /// </remarks>
    public int NumExperts { get; set; } = 8;

    /// <summary>
    /// Gets or sets the number of active experts per token.
    /// </summary>
    /// <value>Defaults to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> For each input token, the router selects this many
    /// experts to process it. Fewer active experts = faster but potentially less accurate.
    /// </para>
    /// </remarks>
    public int NumActiveExperts { get; set; } = 2;

    /// <summary>
    /// Gets or sets the auxiliary loss weight for the router load balancing.
    /// </summary>
    /// <value>Defaults to 0.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This loss encourages the router to distribute tokens
    /// evenly across experts, preventing some experts from being overloaded while others
    /// are idle.
    /// </para>
    /// </remarks>
    public double RouterAuxLossWeight { get; set; } = 0.01;
}
