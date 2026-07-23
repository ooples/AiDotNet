using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DiffusionTS (Interpretable Diffusion for Time Series).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// DiffusionTS is an interpretable diffusion model for time series that uses seasonal-trend
/// decomposition to generate forecasts with clear interpretable components.
/// </para>
/// <para><b>For Beginners:</b> DiffusionTS focuses on making diffusion models more
/// interpretable by decomposing time series into understandable components:
///
/// <b>The Key Insight:</b>
/// Time series often have clear structure (trends, seasonality) that gets lost in
/// "black box" models. DiffusionTS preserves this structure by generating each
/// component separately and combining them.
///
/// <b>How DiffusionTS Works:</b>
/// 1. <b>Decomposition:</b> Split time series into trend, seasonal, and residual
/// 2. <b>Component Diffusion:</b> Generate each component with specialized networks
/// 3. <b>Reconstruction:</b> Combine components to form final forecast
/// 4. <b>Interpretation:</b> Each component has clear meaning
///
/// <b>DiffusionTS Architecture:</b>
/// - Trend Network: Captures long-term movements (slow, smooth)
/// - Seasonal Network: Captures periodic patterns (daily, weekly, yearly)
/// - Residual Network: Captures irregular fluctuations
/// - Fusion Module: Combines components coherently
///
/// <b>Key Benefits:</b>
/// - Interpretable decomposition of forecasts
/// - Can enforce structural constraints (smooth trends, periodic seasons)
/// - Better uncertainty quantification per component
/// - Enables "what-if" analysis by modifying components
/// </para>
/// <para>
/// <b>Reference:</b> Yuan and Qiu, "Diffusion-TS: Interpretable Diffusion for General Time Series Generation", 2024.
/// https://arxiv.org/abs/2403.01742
/// </para>
/// </remarks>
public class DiffusionTSOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DiffusionTSOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default DiffusionTS configuration optimized for
    /// interpretable time series forecasting with decomposition.
    /// </para>
    /// </remarks>
    public DiffusionTSOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DiffusionTSOptions(DiffusionTSOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SequenceLength = other.SequenceLength;
        ForecastHorizon = other.ForecastHorizon;
        NumFeatures = other.NumFeatures;
        HiddenDimension = other.HiddenDimension;
        TrendHiddenDim = other.TrendHiddenDim;
        SeasonalHiddenDim = other.SeasonalHiddenDim;
        NumDiffusionSteps = other.NumDiffusionSteps;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        BetaSchedule = other.BetaSchedule;
        NumSamples = other.NumSamples;
        DecompositionPeriod = other.DecompositionPeriod;
        TrendKernelSize = other.TrendKernelSize;
        UseSeasonalComponent = other.UseSeasonalComponent;
        UseTrendComponent = other.UseTrendComponent;
        DropoutRate = other.DropoutRate;
    }

    /// <summary>
    /// Gets or sets the sequence length (input length).
    /// </summary>
    /// <value>The sequence length, defaulting to 168.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model uses.
    /// Default of 168 corresponds to one week of hourly data.
    /// </para>
    /// </remarks>
    public int SequenceLength { get; set; } = 168;

    /// <summary>
    /// Gets or sets the forecast horizon.
    /// </summary>
    /// <value>The forecast horizon, defaulting to 24.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// Default of 24 corresponds to one day ahead for hourly data.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the number of features.
    /// </summary>
    /// <value>The number of features, defaulting to 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many variables are measured at each time step.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 1;

    /// <summary>
    /// Gets or sets the main hidden dimension.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The base internal representation size.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the hidden dimension for the trend network.
    /// </summary>
    /// <value>The trend hidden dimension, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Trend is typically smooth, so needs less capacity.
    /// Smaller networks prevent overfitting to noise.
    /// </para>
    /// </remarks>
    public int TrendHiddenDim { get; set; } = 32;

    /// <summary>
    /// Gets or sets the hidden dimension for the seasonal network.
    /// </summary>
    /// <value>The seasonal hidden dimension, defaulting to 48.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Seasonality needs moderate capacity to capture
    /// periodic patterns without overfitting.
    /// </para>
    /// </remarks>
    public int SeasonalHiddenDim { get; set; } = 48;

    /// <summary>
    /// Gets or sets the number of diffusion steps.
    /// </summary>
    /// <value>The number of diffusion steps, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More steps = higher quality but slower.
    /// </para>
    /// </remarks>
    public int NumDiffusionSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the starting noise level.
    /// </summary>
    /// <value>The starting beta, defaulting to 0.0001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Initial noise variance in forward diffusion.
    /// </para>
    /// </remarks>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the ending noise level.
    /// </summary>
    /// <value>The ending beta, defaulting to 0.02.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Final noise variance.
    /// </para>
    /// </remarks>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the noise schedule type.
    /// </summary>
    /// <value>The beta schedule, defaulting to "linear".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How noise increases across steps.
    /// "linear" works well for most time series tasks.
    /// </para>
    /// </remarks>
    public string BetaSchedule { get; set; } = "linear";

    /// <summary>
    /// Gets or sets the number of samples for uncertainty estimation.
    /// </summary>
    /// <value>The number of samples, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many forecasts to generate for intervals.
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets the decomposition period for seasonal-trend separation.
    /// </summary>
    /// <value>The decomposition period, defaulting to 24 (daily for hourly data).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The expected periodicity of seasonal patterns
    /// used in the interpretable decomposition. This controls how the model
    /// separates trend from seasonal components.
    /// For hourly data: 24 (daily), 168 (weekly), 8760 (yearly).
    /// For daily data: 7 (weekly), 365 (yearly).
    /// </para>
    /// </remarks>
    public int DecompositionPeriod { get; set; } = 24;

    /// <summary>
    /// Gets or sets the kernel size for trend extraction.
    /// </summary>
    /// <value>The trend kernel size, defaulting to 25.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Moving average window size for extracting trend.
    /// Larger values give smoother trends but may miss shorter-term changes.
    /// Should be odd number for symmetric averaging.
    /// </para>
    /// </remarks>
    public int TrendKernelSize { get; set; } = 25;

    /// <summary>
    /// Gets or sets whether to model seasonal component.
    /// </summary>
    /// <value>True to model seasonality; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Disable if your data has no clear periodic patterns.
    /// </para>
    /// </remarks>
    public bool UseSeasonalComponent { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to model trend component.
    /// </summary>
    /// <value>True to model trend; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Disable if your data has no long-term trend.
    /// </para>
    /// </remarks>
    public bool UseTrendComponent { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Regularization to prevent overfitting.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;
}
