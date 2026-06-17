namespace AiDotNet.Models.Options;

/// <summary>
/// Options for <see cref="AiDotNet.TimeSeries.TiDEModel{T}"/> — Time-series Dense Encoder (Das et al.,
/// "Long-term Forecasting with TiDE: Time-series Dense Encoder", TMLR 2023). A pure-MLP encoder/decoder
/// with a linear residual; on long-horizon benchmarks it matches or beats transformers at a fraction of
/// the cost. Faithful core: a ReLU encoder MLP + decoder projection + a linear skip from the input window.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., <see cref="float"/> or <see cref="double"/>).</typeparam>
/// <remarks>
/// <para>
/// TiDE encodes the lookback window with a multi-layer-perceptron (MLP) encoder into a latent vector of
/// size <see cref="HiddenSize"/>, decodes that to the forecast horizon, and adds a linear residual mapped
/// directly from the input window. Being an MLP, it captures non-linear structure that the purely-linear
/// DLinear/NLinear cannot, while remaining far cheaper than attention-based models.
/// </para>
/// <para><b>For Beginners:</b> this model squeezes the recent history through a small neural network
/// ("encoder") to a compact summary, expands that summary into the forecast ("decoder"), and also adds a
/// simple straight-line shortcut from the input so it never does worse than a linear model. The
/// <see cref="HiddenSize"/> knob sets how big that compact summary is.
/// </para>
/// <para>
/// Typical sizes: <see cref="LookbackWindow"/> 24–96, <see cref="HiddenSize"/> 32–256 (64 is a good
/// default), <see cref="ForecastHorizon"/> 1 for the supervised harness.
/// </para>
/// <para><b>Reference:</b> Abhimanyu Das, Weihao Kong, Andrew Leach, Shaan Mathur, Rajat Sen, Rose Yu.
/// "Long-term Forecasting with TiDE: Time-series Dense Encoder." TMLR 2023. https://arxiv.org/abs/2304.08424
/// </para>
/// </remarks>
public class TiDEOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>History length the encoder sees (input window).</summary>
    /// <value>Number of past time steps fed to the model. Default is 24.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how many recent observations the model looks at. Larger windows
    /// capture longer patterns but need more data; 24 is a good default.</para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 24;

    /// <summary>Forecast horizon. The supervised harness uses 1 (predict the next target value).</summary>
    /// <value>Number of future time steps the model predicts at once. Default is 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how far ahead to forecast. 1 predicts the next step; higher values
    /// predict several steps at once.</para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 1;

    /// <summary>Latent dimension of the MLP encoder (and decoder hidden width).</summary>
    /// <value>Number of hidden units in the encoder/decoder. Default is 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> the size of the compact summary the encoder produces — the model's
    /// capacity knob. Larger values let it learn more complex patterns but increase parameters and the
    /// risk of overfitting on small datasets.</para>
    /// </remarks>
    public int HiddenSize { get; set; } = 64;

    /// <summary>Step size for gradient-descent optimization.</summary>
    /// <value>Default is 0.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how big a step the model takes when updating its weights. Smaller is
    /// slower but more stable; larger trains faster but can overshoot.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>Number of full passes over the training data.</summary>
    /// <value>Default is 50.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how many times the model sees the whole training set. Too many can
    /// lead to overfitting.</para>
    /// </remarks>
    public int Epochs { get; set; } = 50;

    /// <summary>Number of training samples processed per gradient update.</summary>
    /// <value>Default is 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how many examples are averaged before each weight update. Larger
    /// batches are steadier; smaller batches update more often.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>Creates a new <see cref="TiDEOptions{T}"/> with default values.</summary>
    public TiDEOptions() { }

    /// <summary>Creates a deep copy of an existing <see cref="TiDEOptions{T}"/>.</summary>
    /// <param name="other">The options instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public TiDEOptions(TiDEOptions<T> other)
    {
        if (other == null) { throw new ArgumentNullException(nameof(other)); }

        // Inherited properties — TimeSeriesRegressionOptions<T> has no copy constructor, so copy every
        // inherited setting explicitly; otherwise cloning silently resets them to their defaults.
        LagOrder = other.LagOrder;
        IncludeTrend = other.IncludeTrend;
        SeasonalPeriod = other.SeasonalPeriod;
        AutocorrelationCorrection = other.AutocorrelationCorrection;
        ModelType = other.ModelType;
        LossFunction = other.LossFunction;
        MaxPredictionAbsValue = other.MaxPredictionAbsValue;
        MaxTrainingTimeSeconds = other.MaxTrainingTimeSeconds;
        DecompositionMethod = other.DecompositionMethod;
        UseIntercept = other.UseIntercept;
        Seed = other.Seed;

        // TiDE-specific properties.
        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        HiddenSize = other.HiddenSize;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }
}
