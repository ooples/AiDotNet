using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TimeMachine (Time Series State Space Model).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// TimeMachine is a state space model specifically designed for time series forecasting
/// with two outer and two inner Mamba branches.
/// </para>
/// <para><b>For Beginners:</b> TimeMachine is a modern architecture that combines ideas from
/// state space models (like Mamba and S4) with a time-series-specific branch structure.
/// RevIN first normalizes each series. E1 embeds the history length, two outer Mambas
/// scan complementary orientations, E2 forms a smaller embedding for two inner Mambas,
/// and P1/P2 combine residual and concatenated paths into the forecast.
///
/// <b>Advantages:</b>
/// - Linear complexity O(n) from SSM backbone
/// - Complementary scans model both channel and embedding-axis dependencies
/// - Residual paths preserve the original embedded signal
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
    /// the paper's four-Mamba time series forecasting graph.
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

        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
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
    /// <value>The state dimension, defaulting to 256 as in the TimeMachine reference implementation.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The dimension of the hidden state in each SSM.
    /// Larger values capture more complex dynamics but use more memory.
    /// </para>
    /// </remarks>
    public int StateDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the legacy branch-count setting.
    /// </summary>
    /// <value>Four, matching the fixed four-Mamba architecture.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The paper defines exactly four Mamba blocks: two outer
    /// branches and two inner branches. The default native architecture therefore always uses four.
    /// This property remains for compatibility with previously serialized configurations.
    /// </para>
    /// </remarks>
    public int NumScales { get; set; } = 4;

    /// <summary>
    /// Gets or sets the legacy per-scale layer setting.
    /// </summary>
    /// <value>The number of layers per scale, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The paper graph has a fixed depth rather than repeated
    /// per-scale stacks. This property remains for compatibility with older configurations.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the expansion factor for SSM inner dimension.
    /// </summary>
    /// <value>The expansion factor, defaulting to 1 as in the TimeMachine experiments.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each SSM operates in an expanded dimension
    /// (model_dim * expand_factor) for more expressiveness.
    /// </para>
    /// </remarks>
    public int ExpandFactor { get; set; } = 1;

    /// <summary>
    /// Gets or sets the convolution kernel size for local context.
    /// </summary>
    /// <value>The kernel size, defaulting to 2 as in the TimeMachine experiments.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A small 1D convolution captures local patterns
    /// before SSM processing. Typically 3-7.
    /// </para>
    /// </remarks>
    public int ConvKernelSize { get; set; } = 2;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.05 as in the reference implementation.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the legacy multi-scale-attention setting.
    /// </summary>
    /// <value>Retained compatibility value; the paper-faithful default graph does not use attention.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The published TimeMachine architecture combines its
    /// branches with addition and concatenation, not attention. This property remains for
    /// compatibility and does not alter the paper-faithful default graph.
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
    /// Gets or sets the legacy temporal-decomposition label.
    /// </summary>
    /// <value>The retained compatibility label, defaulting to "moving_avg".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Older AiDotNet TimeMachine configurations exposed a
    /// decomposition choice that is not part of the published architecture. The setting
    /// remains so those configurations can still be loaded, but the default graph ignores it.
    /// </para>
    /// </remarks>
    public string TemporalDecompositionMethod { get; set; } = "moving_avg";
}
