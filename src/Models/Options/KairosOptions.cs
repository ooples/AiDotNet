using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Kairos (Adaptive and Generalizable Time Series Foundation Model).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Kairos uses a Mixture-of-Size Encoder with adaptive tokenization that adjusts patch
/// granularity based on local information density. This parameter-efficient approach
/// handles diverse time series characteristics without fixed tokenization.
/// </para>
/// <para><b>For Beginners:</b> Kairos adapts its tokenization to your data:
///
/// <b>Adaptive Tokenization:</b>
/// Unlike fixed-size patching, Kairos uses multiple patch sizes simultaneously and
/// a learned router decides which granularity is best for each segment. Dense/volatile
/// regions get fine-grained tokens; smooth regions get coarse tokens.
///
/// <b>Mixture-of-Size Encoder:</b>
/// Multiple encoder branches process patches at different sizes, then a gating
/// mechanism combines the results based on local information density.
/// </para>
/// <para>
/// <b>Reference:</b> "Kairos: Towards Adaptive and Generalizable Time Series Foundation Models", 2025.
/// https://arxiv.org/abs/2509.25826
/// </para>
/// </remarks>
public class KairosOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public KairosOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public KairosOptions(KairosOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchSizes = (int[])other.PatchSizes.Clone();
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        IntermediateSize = other.IntermediateSize;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
    }

    /// <summary>
    /// Gets or sets the context length.
    /// </summary>
    /// <value>Defaults to 1024.</value>
    public int ContextLength { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the forecast horizon.
    /// </summary>
    /// <value>Defaults to 96.</value>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the multiple patch sizes for adaptive tokenization.
    /// </summary>
    /// <value>Defaults to [8, 16, 32, 64].</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Kairos processes input at multiple granularities
    /// simultaneously. The router learns which granularity is best for each region.
    /// </para>
    /// </remarks>
    public int[] PatchSizes { get; set; } = [8, 16, 32, 64];

    /// <summary>
    /// Gets or sets the hidden dimension.
    /// </summary>
    /// <value>Defaults to 512.</value>
    public int HiddenDimension { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Defaults to 12.</value>
    public int NumLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 8.</value>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the intermediate size for the feed-forward network.
    /// </summary>
    /// <value>Defaults to 2048.</value>
    public int IntermediateSize { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>Defaults to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>Defaults to <see cref="FoundationModelSize.Base"/>.</value>
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;
}
