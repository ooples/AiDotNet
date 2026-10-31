using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the DeepAR (Deep Autoregressive) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// DeepAR is a probabilistic forecasting methodology based on autoregressive recurrent neural networks.
/// Unlike traditional methods that provide point forecasts, DeepAR produces probabilistic forecasts
/// that include prediction intervals. It's particularly effective for:
/// - Handling multiple related time series simultaneously
/// - Cold-start problems (forecasting for new items with limited history)
/// - Capturing complex seasonal patterns and trends
/// - Quantifying forecast uncertainty
/// </para>
/// <para><b>For Beginners:</b> DeepAR is an advanced forecasting model that not only predicts
/// what will happen, but also how confident it is in those predictions. Instead of saying
/// "sales will be exactly 100 units," it might say "sales will likely be between 80 and 120 units,
/// with 100 being most probable."
///
/// This is especially useful when:
/// - You need to plan for worst-case and best-case scenarios
/// - You have many related time series (e.g., sales across many stores)
/// - You have some series with very little historical data
///
/// The "autoregressive" part means it uses its own predictions as inputs for future predictions,
/// and "deep" refers to the use of deep neural networks (specifically, LSTM networks).
/// </para>
/// </remarks>
public class DeepAROptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DeepAROptions{T}"/> class.
    /// </summary>
    public DeepAROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public DeepAROptions(DeepAROptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        HiddenSize = other.HiddenSize;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
        NumSamples = other.NumSamples;
        LikelihoodType = other.LikelihoodType;
        CovariateSize = other.CovariateSize;
        EmbeddingDimension = other.EmbeddingDimension;
        ScaleHandling = other.ScaleHandling;
        UseAgeFeature = other.UseAgeFeature;
    }

    /// <summary>
    /// Gets or sets the conditioning range t₀: the number of past steps that set the scale and the state before a
    /// forecast (the paper's encoder length, which it sets per dataset: 8 to 168).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model looks at before
    /// making a prediction. For daily data, 30 would mean looking at the past month.
    /// </para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 30;

    /// <summary>
    /// Gets or sets the prediction range, the number of steps forecast by ancestral sampling (the paper's decoder
    /// length, 8 to 52 per dataset).
    /// </summary>
    public int ForecastHorizon { get; set; } = 7;

    /// <summary>
    /// Gets or sets the number of cells per LSTM layer (the paper's Table 3: 40, or 120 on the large datasets).
    /// </summary>
    public int HiddenSize { get; set; } = 40;

    /// <summary>
    /// Gets or sets the number of LSTM layers (Table 3: 3).
    /// </summary>
    public int NumLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the dropout between LSTM layers (the paper is silent; GluonTS's default 0.1).
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the Adam learning rate (Table 3: 1e-3).
    /// </summary>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of training epochs over all windows.
    /// </summary>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the batch size (Table 3: 64).
    /// </summary>
    public int BatchSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of ancestral sample paths a forecast draws (the paper's appendix: 200).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model forecasts by simulating many possible futures; the point forecast is
    /// their median and the spread gives the prediction intervals.</para>
    /// </remarks>
    public int NumSamples { get; set; } = 200;

    /// <summary>
    /// Gets or sets the likelihood: "Gaussian" (real-valued data) or "NegativeBinomial" (counts), the paper's two, or
    /// "StudentT" (heavy tails; the GluonTS reference's default, its degrees of freedom learned).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This determines what kind of randomness the model assumes in your data: Gaussian
    /// for temperatures or prices, negative binomial when counting things, Student-t when occasional extreme
    /// values occur.</para>
    /// </remarks>
    public string LikelihoodType { get; set; } = "Gaussian";

    /// <summary>
    /// Gets or sets the number of covariates per step (the time-series model reads them from the input matrix; the
    /// paper standardizes every covariate).
    /// </summary>
    public int CovariateSize { get; set; } = 0;

    /// <summary>
    /// Gets or sets the size of the learned item embedding when several series are modelled together (§3.4: a
    /// categorical item feature with a learned embedding; Table 3: 20 on the larger datasets).
    /// </summary>
    public int EmbeddingDimension { get; set; } = 20;

    /// <summary>
    /// Gets or sets how the conditioning range is rescaled before the recurrent trunk sees it.
    /// </summary>
    /// <value>The scale-handling rule, defaulting to the paper's <c>1 + mean(z)</c>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Series arrive on very different magnitudes. DeepAR divides each
    /// one by a single number summarizing its own level, learns on the result, and multiplies the
    /// prediction back, so the network is not spending its capacity on magnitudes.
    /// </para>
    /// <para>
    /// Salinas et al. 2020, section 3.3 defines that number as <c>nu = 1 + mean(z)</c>, which is
    /// the default here. That definition assumes non-negative data, so a signed series centered on
    /// zero - a return series, say - is better served by
    /// <see cref="DeepARScaleHandling.MeanAbsolute"/>, and data the caller has already normalized
    /// by <see cref="DeepARScaleHandling.None"/>.
    /// </para>
    /// </remarks>
    public DeepARScaleHandling ScaleHandling { get; set; } = DeepARScaleHandling.PaperMean;

    /// <summary>
    /// Gets or sets whether the time-series model adds the paper's "age" covariate, the distance to the series' first
    /// observation, standardized (§3.4: used in all experiments).
    /// </summary>
    public bool UseAgeFeature { get; set; } = true;

    /// <summary>The likelihood named by <see cref="LikelihoodType"/>.</summary>
    /// <exception cref="ArgumentException">For any other name.</exception>
    public AiDotNet.TimeSeries.DeepARLikelihood ResolveLikelihood()
    {
        if (Enum.TryParse<AiDotNet.TimeSeries.DeepARLikelihood>(LikelihoodType?.Trim(), ignoreCase: true, out var likelihood)
            && Enum.IsDefined(typeof(AiDotNet.TimeSeries.DeepARLikelihood), likelihood))
            return likelihood;
        throw new ArgumentException($"Unknown DeepAR likelihood \"{LikelihoodType}\"; use Gaussian, NegativeBinomial or StudentT.");
    }
}
