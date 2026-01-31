using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TimeMachine (Time Series State Space Model).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// TimeMachine is a state space model specifically designed for time series forecasting
/// that combines the efficiency of SSMs with specialized temporal modeling components.
/// </para>
/// <para><b>For Beginners:</b> TimeMachine is a modern architecture that combines ideas from
/// state space models (like Mamba and S4) with time series-specific enhancements:
///
/// <b>The Key Insight:</b>
/// While Mamba and S4 are general-purpose sequence models, TimeMachine is specifically
/// designed for time series data with features like:
/// 1. Multi-scale temporal decomposition
/// 2. Trend-seasonality modeling
/// 3. Efficient long-range dependency capture
///
/// <b>How It Works:</b>
/// 1. <b>Temporal Decomposition:</b> Separates trend, seasonal, and residual components
/// 2. <b>Multi-Scale SSM:</b> Processes different temporal scales with dedicated SSM blocks
/// 3. <b>Adaptive Gating:</b> Learns which scales are most important for each prediction
/// 4. <b>Reconstruction:</b> Combines multi-scale outputs for final forecast
///
/// <b>Architecture:</b>
/// - Input embedding with reversible instance normalization
/// - Multi-scale SSM blocks (fine, medium, coarse granularity)
/// - Scale-wise attention for importance weighting
/// - Output projection with de-normalization
///
/// <b>Advantages:</b>
/// - Linear complexity O(n) from SSM backbone
/// - Explicit temporal decomposition improves interpretability
/// - Multi-scale processing captures patterns at different frequencies
/// - State-of-the-art results on time series benchmarks
/// </para>
/// <para>
/// <b>Reference:</b> Ahamed et al., "TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting", 2024.
/// https://arxiv.org/abs/2403.09898
/// </para>
/// </remarks>
public class TimeMachineOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TimeMachineOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default TimeMachine configuration optimized for
    /// multi-scale time series forecasting with efficient state space processing.
    /// </para>
    /// </remarks>
    public TimeMachineOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TimeMachineOptions(TimeMachineOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        ModelDimension = other.ModelDimension;
        StateDimension = other.StateDimension;
        NumScales = other.NumScales;
        NumLayers = other.NumLayers;
        ExpandFactor = other.ExpandFactor;
        ConvKernelSize = other.ConvKernelSize;
        DropoutRate = other.DropoutRate;
        UseMultiScaleAttention = other.UseMultiScaleAttention;
        UseReversibleNormalization = other.UseReversibleNormalization;
        TemporalDecompositionMethod = other.TemporalDecompositionMethod;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>The context length, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at.
    /// TimeMachine handles long contexts efficiently via SSM backbone.
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
    /// Gets or sets the state dimension for each SSM block.
    /// </summary>
    /// <value>The state dimension, defaulting to 16.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The dimension of the hidden state in each SSM.
    /// Larger values capture more complex dynamics but use more memory.
    /// </para>
    /// </remarks>
    public int StateDimension { get; set; } = 16;

    /// <summary>
    /// Gets or sets the number of temporal scales to model.
    /// </summary>
    /// <value>The number of scales, defaulting to 4 (as per "4 Mambas" in the paper).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> TimeMachine uses multiple SSM blocks at different
    /// temporal granularities. The default of 4 corresponds to:
    /// - Scale 1: Fine-grained (captures high-frequency patterns)
    /// - Scale 2: Medium-fine (captures daily patterns)
    /// - Scale 3: Medium-coarse (captures weekly patterns)
    /// - Scale 4: Coarse (captures long-term trends)
    /// More scales increase capacity but also computation.
    /// </para>
    /// </remarks>
    public int NumScales { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of SSM layers per scale.
    /// </summary>
    /// <value>The number of layers per scale, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many SSM blocks are stacked at each scale.
    /// More layers = deeper model per scale = more capacity.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the expansion factor for SSM inner dimension.
    /// </summary>
    /// <value>The expansion factor, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each SSM operates in an expanded dimension
    /// (model_dim * expand_factor) for more expressiveness.
    /// </para>
    /// </remarks>
    public int ExpandFactor { get; set; } = 2;

    /// <summary>
    /// Gets or sets the convolution kernel size for local context.
    /// </summary>
    /// <value>The kernel size, defaulting to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A small 1D convolution captures local patterns
    /// before SSM processing. Typically 3-7.
    /// </para>
    /// </remarks>
    public int ConvKernelSize { get; set; } = 4;

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
    /// Gets or sets whether to use multi-scale attention for combining scales.
    /// </summary>
    /// <value>True to use attention for scale combination; false for simple concatenation. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When true, the model learns to weight different scales
    /// dynamically based on the input. When false, scales are simply concatenated.
    /// Attention provides more flexibility but adds computation.
    /// </para>
    /// </remarks>
    public bool UseMultiScaleAttention { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use reversible instance normalization.
    /// </summary>
    /// <value>True for reversible normalization; false for standard. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Reversible instance normalization (RevIN) normalizes
    /// each time series individually and can reverse the normalization after prediction.
    /// This helps handle non-stationary time series with varying scales and trends.
    /// </para>
    /// </remarks>
    public bool UseReversibleNormalization { get; set; } = true;

    /// <summary>
    /// Gets or sets the temporal decomposition method for multi-scale processing.
    /// </summary>
    /// <value>The temporal decomposition method, defaulting to "moving_avg".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How to decompose the time series into components:
    /// - "moving_avg": Uses moving average to extract trend (fast, simple)
    /// - "fft": Uses FFT for frequency-based decomposition (better for periodic data)
    /// - "learnable": Learns the decomposition (most flexible but more parameters)
    /// </para>
    /// </remarks>
    public string TemporalDecompositionMethod { get; set; } = "moving_avg";
}
