using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TimeGrad (Autoregressive Denoising Diffusion Model for Time Series).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// TimeGrad is a probabilistic time series forecasting model that uses denoising diffusion
/// to generate accurate forecasts with well-calibrated uncertainty estimates.
/// </para>
/// <para><b>For Beginners:</b> TimeGrad brings the power of diffusion models (like those
/// used in image generation) to time series forecasting:
///
/// <b>The Key Insight:</b>
/// Most forecasting models give you ONE prediction. But in practice, you want to know
/// "how uncertain is this prediction?" TimeGrad solves this by modeling the FULL probability
/// distribution of future values using a diffusion process.
///
/// <b>How Diffusion Works (simplified):</b>
/// 1. <b>Forward Process:</b> Gradually add noise to data until it becomes pure noise
/// 2. <b>Reverse Process:</b> Learn to remove noise step-by-step, generating samples
/// 3. <b>Conditioning:</b> Use historical data to guide the denoising
/// 4. <b>Sampling:</b> Generate multiple forecasts from the learned distribution
///
/// <b>TimeGrad Architecture:</b>
/// - RNN encoder processes historical data
/// - Diffusion model generates future values conditioned on hidden state
/// - Multiple samples give uncertainty estimates
///
/// <b>Key Benefits:</b>
/// - Probabilistic forecasts (not just point predictions)
/// - Well-calibrated uncertainty estimates
/// - Can generate diverse forecast scenarios
/// - State-of-the-art accuracy on probabilistic metrics
/// </para>
/// <para>
/// <b>Reference:</b> Rasul et al., "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting", 2021.
/// https://arxiv.org/abs/2101.12072
/// </para>
/// </remarks>
public class TimeGradOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TimeGradOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default TimeGrad configuration optimized for
    /// probabilistic time series forecasting.
    /// </para>
    /// </remarks>
    public TimeGradOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TimeGradOptions(TimeGradOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension;
        NumRnnLayers = other.NumRnnLayers;
        NumDiffusionSteps = other.NumDiffusionSteps;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        BetaSchedule = other.BetaSchedule;
        NumSamples = other.NumSamples;
        DropoutRate = other.DropoutRate;
        UseResidualConnection = other.UseResidualConnection;
        DenoisingNetworkDim = other.DenoisingNetworkDim;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>The context length, defaulting to 168 (one week of hourly data).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at
    /// to make predictions. The default of 168 corresponds to one week of hourly data.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 168;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 24 (one day ahead for hourly data).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// The default of 24 corresponds to one day ahead for hourly data.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the hidden dimension for the RNN encoder.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the RNN hidden state that encodes
    /// historical information. Larger values capture more complex patterns.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of RNN layers in the encoder.
    /// </summary>
    /// <value>The number of RNN layers, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many RNN layers are stacked to encode
    /// historical data. More layers = more capacity but also more computation.
    /// </para>
    /// </remarks>
    public int NumRnnLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of diffusion steps (T in the paper).
    /// </summary>
    /// <value>The number of diffusion steps, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The diffusion process adds noise in T steps.
    /// More steps = finer-grained denoising = better quality but slower.
    ///
    /// <b>Analogy:</b> Like slowly adding static to a TV signal (forward process),
    /// then learning to remove it frame-by-frame (reverse process).
    /// </para>
    /// </remarks>
    public int NumDiffusionSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the starting noise level (beta_1).
    /// </summary>
    /// <value>The starting beta, defaulting to 0.0001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much noise to add at the first diffusion step.
    /// Small values mean the first step barely changes the data.
    /// </para>
    /// </remarks>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the ending noise level (beta_T).
    /// </summary>
    /// <value>The ending beta, defaulting to 0.02.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much noise to add at the last diffusion step.
    /// By step T, the data should be almost completely noise.
    /// </para>
    /// </remarks>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the noise schedule type.
    /// </summary>
    /// <value>The beta schedule, defaulting to "linear".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How noise levels change across diffusion steps:
    /// - "linear": Noise increases uniformly from beta_start to beta_end
    /// - "cosine": Smoother schedule, often better for images
    /// - "quadratic": Noise increases quadratically
    /// </para>
    /// </remarks>
    public string BetaSchedule { get; set; } = "linear";

    /// <summary>
    /// Gets or sets the number of samples to generate for probabilistic forecasting.
    /// </summary>
    /// <value>The number of samples, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different forecast paths to generate.
    /// More samples = better uncertainty estimates but slower inference.
    ///
    /// <b>Example:</b> With 100 samples, you can compute the 10th and 90th
    /// percentiles to get an 80% prediction interval.
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting during training.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use residual connections in the denoising network.
    /// </summary>
    /// <value>True to use residual connections; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Residual connections help gradients flow during
    /// training by adding shortcut paths. Generally improves training stability.
    /// </para>
    /// </remarks>
    public bool UseResidualConnection { get; set; } = true;

    /// <summary>
    /// Gets or sets the dimension of the denoising network.
    /// </summary>
    /// <value>The denoising network dimension, defaulting to 128.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The hidden size of the network that learns to
    /// remove noise. Larger values can model more complex noise patterns.
    /// </para>
    /// </remarks>
    public int DenoisingNetworkDim { get; set; } = 128;
}
