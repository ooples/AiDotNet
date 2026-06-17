namespace AiDotNet.Models.Options;

/// <summary>
/// Options for <see cref="AiDotNet.TimeSeries.DLinearModel{T}"/> — the decomposition-linear forecaster
/// (Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023). Despite its
/// simplicity it is a strong, current baseline that frequently matches or beats heavier transformers on
/// standard long-horizon benchmarks, at a fraction of the cost.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., <see cref="float"/> or <see cref="double"/>).</typeparam>
/// <remarks>
/// <para>
/// DLinear decomposes each input window into a trend component (a moving average) and the remaining
/// seasonal component, applies one independent linear map to each, and sums the two projections to
/// produce the forecast. The whole model is two linear layers — there is no attention and no recurrence —
/// so it trains in seconds and is a natural sanity-check baseline before reaching for a transformer.
/// </para>
/// <para><b>For Beginners:</b> this model splits a series into its slow-moving "trend" (the moving
/// average) and the leftover wiggles ("seasonal"), learns a simple straight-line mapping for each, and
/// adds them back together to predict the future. It is fast, hard to overfit, and surprisingly accurate,
/// which makes it an excellent first model to try and a yardstick for judging fancier ones.
/// </para>
/// <para>
/// Typical sizes: a <see cref="LookbackWindow"/> of 24–96 steps with <see cref="ForecastHorizon"/> 1
/// (the supervised harness predicts the next value); the parameter count is roughly
/// <c>2 × LookbackWindow × ForecastHorizon</c>, i.e. a few thousand parameters even for long windows.
/// </para>
/// <para><b>Reference:</b> Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu. "Are Transformers Effective for
/// Time Series Forecasting?" AAAI 2023. https://arxiv.org/abs/2205.13504
/// </para>
/// </remarks>
public class DLinearOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>History length the linear maps see (input window).</summary>
    /// <value>Number of past time steps fed to the model. Default is 24.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how many recent observations the model looks at to make a prediction.
    /// A larger window lets it capture longer patterns (e.g., weekly cycles) but needs more data and
    /// trains a touch slower. 24 is a sensible default for hourly/daily series.</para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 24;

    /// <summary>Forecast horizon. The supervised harness uses 1 (predict the next target value).</summary>
    /// <value>Number of future time steps the model predicts at once. Default is 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how far ahead to forecast. 1 means "predict the next step"; set it
    /// higher to predict several steps at once (multi-horizon forecasting).</para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 1;

    /// <summary>Moving-average kernel for the trend/seasonal decomposition (odd; clamped to the window).</summary>
    /// <value>Window size of the moving average that extracts the trend. Default is 25.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> the smoothing width used to separate the slow "trend" from the fast
    /// "seasonal" wiggles. A larger kernel produces a smoother trend; it should be odd and is automatically
    /// clamped so it never exceeds the lookback window.</para>
    /// </remarks>
    public int MovingAverageKernel { get; set; } = 25;

    /// <summary>Step size for gradient-descent optimization.</summary>
    /// <value>Default is 0.01, a common DLinear training configuration.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> controls how big a step the model takes when updating its weights after
    /// each batch. Smaller values (e.g., 0.001) train more slowly but stably; larger values (e.g., 0.1)
    /// train faster but can overshoot and diverge.</para>
    /// <para><b>Default source:</b> general-purpose baseline for this library's supervised forecasting
    /// harness. The LTSF-Linear reference implementation (Zeng et al., AAAI 2023) tunes the learning rate
    /// per dataset (commonly 1e-3 to 5e-2) rather than fixing a single value, so 0.01 is a safe default,
    /// not a paper-specified constant.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>Number of full passes over the training data.</summary>
    /// <value>Default is 50.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how many times the model sees the whole training set. More epochs can
    /// improve the fit up to a point, after which the model starts memorizing noise (overfitting).</para>
    /// <para><b>Default source:</b> library baseline. The reference implementation uses early stopping
    /// rather than a fixed epoch budget; 50 is a reasonable cap for this library's supervised harness.</para>
    /// </remarks>
    public int Epochs { get; set; } = 50;

    /// <summary>Number of training samples processed per gradient update.</summary>
    /// <value>Default is 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> how many examples are averaged together before each weight update.
    /// Larger batches give steadier but less frequent updates; smaller batches are noisier but update
    /// more often.</para>
    /// <para><b>Default source:</b> a common minibatch size and this library's baseline; the LTSF-Linear
    /// reference implementation uses dataset-dependent batch sizes rather than a single fixed value.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>Creates a new <see cref="DLinearOptions{T}"/> with default values.</summary>
    public DLinearOptions() { }

    /// <summary>Creates a deep copy of an existing <see cref="DLinearOptions{T}"/>.</summary>
    /// <param name="other">The options instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public DLinearOptions(DLinearOptions<T> other)
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

        // DLinear-specific properties.
        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        MovingAverageKernel = other.MovingAverageKernel;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }
}
