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
/// <para>
/// <b>MIGRATION — four properties were removed:</b> <c>NumScales</c>, <c>NumLayers</c>,
/// <c>UseMultiScaleAttention</c> and <c>TemporalDecompositionMethod</c>. Each was public, settable
/// and read by nothing. The published graph is fixed — exactly four Mambas in the two-outer /
/// two-inner arrangement, combined by addition and concatenation rather than attention — so setting
/// any of them produced the identical model with no error and no signal. Delete the assignments;
/// there is no replacement, because there was never a behavior to replace. A build error naming the
/// property is the point: it tells you the setting was doing nothing, which the old code did not.
/// The dimensions that DO change the model — <see cref="ModelDimension"/>,
/// <see cref="StateDimension"/>, <see cref="ExpandFactor"/>, <see cref="ConvKernelSize"/>,
/// <see cref="ContextLength"/>, <see cref="ForecastHorizon"/> and <see cref="DropoutRate"/> —
/// are unchanged.
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
        ExpandFactor = other.ExpandFactor;
        ConvKernelSize = other.ConvKernelSize;
        DropoutRate = other.DropoutRate;
        UseReversibleNormalization = other.UseReversibleNormalization;
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

}
