using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Mamba (Selective State Space Model).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Mamba is a selective state space model that achieves linear-time complexity for
/// sequence modeling while maintaining the expressiveness of transformers through
/// input-dependent (selective) state space parameters.
/// </para>
/// <para><b>For Beginners:</b> Mamba is a breakthrough in efficient sequence modeling:
///
/// <b>The Key Insight:</b>
/// Transformers have O(n^2) complexity due to attention, which is slow for long sequences.
/// State space models (SSMs) have O(n) complexity but are less expressive.
/// Mamba makes SSM parameters input-dependent (selective), combining the best of both.
///
/// <b>How It Works:</b>
/// 1. <b>State Space Model:</b> Maintains a hidden state updated recurrently
/// 2. <b>Selective Mechanism:</b> Parameters (A, B, C, delta) vary with input
/// 3. <b>Hardware-aware Algorithm:</b> Efficient implementation via parallel scan
/// 4. <b>Linear Complexity:</b> O(n) time and memory for sequence length n
///
/// <b>Architecture:</b>
/// - Input projection to expanded dimension
/// - 1D convolution for local context
/// - Selective SSM core with input-dependent parameters
/// - Output projection back to model dimension
///
/// <b>Advantages:</b>
/// - Linear time complexity (vs O(n^2) for attention)
/// - Handles very long sequences efficiently
/// - Strong performance on language, audio, and time series
/// - Hardware-efficient implementation
/// </para>
/// <para>
/// <b>Reference:</b> Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024.
/// https://arxiv.org/abs/2312.00752
/// </para>
/// </remarks>
public class MambaOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MambaOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default Mamba configuration for
    /// efficient linear-time sequence modeling.
    /// </para>
    /// </remarks>
    public MambaOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MambaOptions(MambaOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        ModelDimension = other.ModelDimension;
        StateDimension = other.StateDimension;
        ExpandFactor = other.ExpandFactor;
        ConvKernelSize = other.ConvKernelSize;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        DtRank = other.DtRank;
        UseBidirectional = other.UseBidirectional;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>The context length, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at.
    /// Mamba handles long contexts efficiently due to linear complexity.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the model dimension (d_model).
    /// </summary>
    /// <value>The model dimension, defaulting to 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The main hidden dimension of the model.
    /// Controls the capacity for learning patterns.
    /// </para>
    /// </remarks>
    public int ModelDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the state dimension (d_state or N).
    /// </summary>
    /// <value>The state dimension, defaulting to 16.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The dimension of the hidden state in the SSM.
    /// Larger values can capture more complex dynamics but use more memory.
    /// </para>
    /// </remarks>
    public int StateDimension { get; set; } = 16;

    /// <summary>
    /// Gets or sets the expansion factor for the inner dimension.
    /// </summary>
    /// <value>The expansion factor, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The SSM operates in an expanded dimension
    /// (model_dim * expand_factor) for more expressiveness.
    /// </para>
    /// </remarks>
    public int ExpandFactor { get; set; } = 2;

    /// <summary>
    /// Gets or sets the convolution kernel size.
    /// </summary>
    /// <value>The kernel size, defaulting to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A small 1D convolution is applied before the SSM
    /// to capture local patterns. This is typically 3-7.
    /// </para>
    /// </remarks>
    public int ConvKernelSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of Mamba layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many Mamba blocks are stacked.
    /// More layers = deeper model = more capacity.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the rank for the delta (dt) projection.
    /// </summary>
    /// <value>The dt rank, defaulting to "auto" (ceil(d_model/16)).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The delta parameter controls how much new input
    /// affects the hidden state. This is its projection rank.
    /// Use -1 for automatic calculation (d_model / 16).
    /// </para>
    /// </remarks>
    public int DtRank { get; set; } = -1; // -1 means auto (d_model / 16)

    /// <summary>
    /// Gets or sets whether to use bidirectional processing.
    /// </summary>
    /// <value>True for bidirectional; false for causal/unidirectional. Default: false.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Bidirectional Mamba processes the sequence both
    /// forward and backward, which can improve accuracy but doubles computation.
    /// Use unidirectional (false) for autoregressive tasks like forecasting.
    /// </para>
    /// </remarks>
    public bool UseBidirectional { get; set; } = false;
}
