using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for FlowState (IBM's SSM-based Time Series Foundation Model).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FlowState is IBM's State-Space Model (SSM) based time series foundation model with only
/// 9.1M parameters. Despite being the smallest model in the GIFT-Eval top 10, it outperforms
/// models 20x its size and generalizes to unseen timescales.
/// </para>
/// <para><b>For Beginners:</b> FlowState proves bigger isn't always better:
///
/// <b>State-Space Model Architecture:</b>
/// Instead of attention (which is quadratic in sequence length), FlowState uses
/// structured state spaces (like S4/Mamba) that are linear in sequence length.
/// This makes it extremely efficient for long sequences.
///
/// <b>Key Advantages:</b>
/// - Only 9.1M parameters (smallest in GIFT-Eval top 10)
/// - Outperforms models 20x its size
/// - Generalizes to unseen timescales
/// - Linear-time computation for long sequences
/// </para>
/// <para>
/// <b>Reference:</b> IBM Research, "SSM Time Series Model", 2025.
/// https://research.ibm.com/blog/SSM-time-series-model
/// </para>
/// </remarks>
public class FlowStateOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public FlowStateOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public FlowStateOptions(FlowStateOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        StateDimension = other.StateDimension;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        SSMRank = other.SSMRank;
        UseDiscretization = other.UseDiscretization;
    }

    /// <summary>
    /// Gets or sets the context length.
    /// </summary>
    /// <value>Defaults to 2048.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> SSMs handle long sequences efficiently, so FlowState
    /// supports very long contexts. 2048 is a good default for most applications.</para>
    /// </remarks>
    public int ContextLength { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the forecast horizon.
    /// </summary>
    /// <value>Defaults to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many future steps to predict.</para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the state dimension for the SSM.
    /// </summary>
    /// <value>Defaults to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The SSM maintains a hidden state of this dimension.
    /// Larger state = more memory of past inputs, but more computation.
    /// </para>
    /// </remarks>
    public int StateDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the hidden dimension.
    /// </summary>
    /// <value>Defaults to 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls the model's representational capacity. FlowState
    /// achieves strong results with small dimensions due to the efficiency of SSMs.</para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of SSM layers.
    /// </summary>
    /// <value>Defaults to 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each SSM layer processes the sequence with a learned state-space
    /// model. More layers allow deeper temporal reasoning.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 8;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>Defaults to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Regularization to prevent overfitting.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>Defaults to <see cref="FoundationModelSize.Small"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> FlowState defaults to Small (9.1M params) because
    /// it achieves competitive results at this size.</para>
    /// </remarks>
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Small;

    /// <summary>
    /// Gets or sets the SSM rank for low-rank parameterization.
    /// </summary>
    /// <value>Defaults to 16.</value>
    public int SSMRank { get; set; } = 16;

    /// <summary>
    /// Gets or sets whether to use discretization for continuous-time SSM.
    /// </summary>
    /// <value>Defaults to true.</value>
    public bool UseDiscretization { get; set; } = true;
}
