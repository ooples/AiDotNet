namespace AiDotNet.Models.Options;

/// <summary>
/// Options for <see cref="AiDotNet.TimeSeries.NLinearModel{T}"/> — the normalization-linear forecaster
/// (Zeng et al., AAAI 2023). It subtracts the last observed value from the window, applies a single linear
/// map, then adds the value back — a distribution-shift-robust baseline that, like DLinear, is a strong
/// modern control against heavier models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., <see cref="float"/> or <see cref="double"/>).</typeparam>
/// <remarks>
/// <para>
/// NLinear is the simplest member of the LTSF-Linear family: normalize the input window by subtracting
/// its last value, run one linear layer, then add the last value back to the output. That subtract/add
/// step makes the model robust to distribution shift between training and test (a common failure mode of
/// transformers on long-horizon forecasting), while keeping the model to a single linear map.
/// </para>
/// <para><b>For Beginners:</b> this model first "re-centers" the recent history around its most recent
/// point, learns a straight-line mapping to the future, then shifts the prediction back up by that same
/// recent value. The re-centering trick helps a lot when the overall level of the series drifts over time.
/// </para>
/// <para>
/// Typical sizes: a <see cref="LookbackWindow"/> of 24–96 steps with <see cref="ForecastHorizon"/> 1; the
/// parameter count is about <c>LookbackWindow × ForecastHorizon</c> — a few thousand weights at most.
/// </para>
/// <para><b>Reference:</b> Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu. "Are Transformers Effective for
/// Time Series Forecasting?" AAAI 2023. https://arxiv.org/abs/2205.13504
/// </para>
/// </remarks>
public class NLinearOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>History length the linear map sees (input window).</summary>
    /// <value>Number of past time steps fed to the model. Default is 24.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how many recent observations the model looks at to make a prediction.
    /// Larger windows capture longer patterns but need more data; 24 is a good default.</para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 24;

    /// <summary>Forecast horizon. The supervised harness uses 1 (predict the next target value).</summary>
    /// <value>Number of future time steps the model predicts at once. Default is 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how far ahead to forecast. 1 predicts the next step; higher values
    /// predict several steps at once.</para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 1;

    /// <summary>Step size for gradient-descent optimization.</summary>
    /// <value>Default is 0.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how big a step the model takes when updating its weights. Smaller is
    /// slower but more stable; larger trains faster but can overshoot.</para>
    /// <para><b>Default source:</b> general-purpose baseline for this library's supervised forecasting
    /// harness. The LTSF-Linear reference implementation (Zeng et al., AAAI 2023) tunes the learning rate
    /// per dataset (commonly 1e-3 to 5e-2) rather than fixing a single value, so 0.01 is a safe default,
    /// not a paper-specified constant.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>Number of full passes over the training data.</summary>
    /// <value>Default is 50.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how many times the model sees the whole training set. Too many can
    /// lead to overfitting.</para>
    /// <para><b>Default source:</b> library baseline. The reference implementation uses early stopping
    /// rather than a fixed epoch budget; 50 is a reasonable cap for this library's supervised harness.</para>
    /// </remarks>
    public int Epochs { get; set; } = 50;

    /// <summary>Number of training samples processed per gradient update.</summary>
    /// <value>Default is 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how many examples are averaged before each weight update. Larger
    /// batches are steadier; smaller batches update more often.</para>
    /// <para><b>Default source:</b> a common minibatch size and this library's baseline; the LTSF-Linear
    /// reference implementation uses dataset-dependent batch sizes rather than a single fixed value.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>Creates a new <see cref="NLinearOptions{T}"/> with default values.</summary>
    public NLinearOptions() { }

    /// <summary>Creates a deep copy of an existing <see cref="NLinearOptions{T}"/>.</summary>
    /// <param name="other">The options instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public NLinearOptions(NLinearOptions<T> other)
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

        // NLinear-specific properties.
        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }
}
