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
    /// <summary>Creates options with the paper's defaults.</summary>
    public TimeGradOptions()
    {
        // Seed is INHERITED from ModelOptions and assigned here rather than shadowed with `new`,
        // so every ModelOptions reader sees the same value. A default makes inference reproducible
        // out of the box; callers who want fresh draws per call can set it back to null.
        Seed = 1;
    }

    /// <summary>Copies every setting of <paramref name="other"/>.</summary>
    public TimeGradOptions(TimeGradOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
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
        DenoisingNetworkDim = other.DenoisingNetworkDim;
        ResidualLayers = other.ResidualLayers;
        ResidualChannels = other.ResidualChannels;
        DilationCycleLength = other.DilationCycleLength;
        TimeEmbeddingDim = other.TimeEmbeddingDim;
    }

    /// <summary>
    /// Gets or sets the number of past steps the model conditions on.
    /// </summary>
    /// <value>The context length in time steps; 168 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How much history the model reads before forecasting. With hourly data, 168 is one week.
    /// </para>
    /// <para>
    /// Default: 168, one week of hourly observations, as in the paper's electricity and traffic experiments (Rasul et al. 2021, Sec. 4).
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 168;

    /// <summary>
    /// Gets or sets the number of future steps to forecast.
    /// </summary>
    /// <value>The forecast horizon in time steps; 24 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How far ahead the model predicts. With hourly data, 24 is one day.
    /// </para>
    /// <para>
    /// Default: 24, the paper's day-ahead horizon (Rasul et al. 2021, Sec. 4).
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the hidden size of each RNN layer in the history encoder.
    /// </summary>
    /// <value>The LSTM width; 40 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How much the model can remember about the history at each step. Larger values capture more but train slower.
    /// </para>
    /// <para>
    /// Default: 40, the paper's RNN size (Rasul et al. 2021, Sec. 4).
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 40;

    /// <summary>
    /// Gets or sets the number of stacked LSTM layers in the history encoder.
    /// </summary>
    /// <value>The LSTM depth; 2 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many recurrent layers read the history one after another.
    /// </para>
    /// <para>
    /// Default: 2, the paper's encoder depth (Rasul et al. 2021, Sec. 4).
    /// </para>
    /// </remarks>
    public int NumRnnLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of diffusion steps N.
    /// </summary>
    /// <value>The length of the noising chain; 100 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many small denoising steps turn random noise into a forecast. More steps give finer samples but cost more time per forecast.
    /// </para>
    /// <para>
    /// Default: 100, the paper's N (Rasul et al. 2021, Sec. 4).
    /// </para>
    /// </remarks>
    public int NumDiffusionSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the first noise variance beta_1.
    /// </summary>
    /// <value>A variance in (0, 1); 1e-4 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How much noise the first diffusion step adds. It is tiny so the earliest step barely changes the data.
    /// </para>
    /// <para>
    /// Default: 1e-4, the paper's beta_1 (Rasul et al. 2021, Sec. 4).
    /// </para>
    /// </remarks>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the last noise variance beta_N.
    /// </summary>
    /// <value>A variance in (0, 1); 0.1 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How much noise the last diffusion step adds. It is large so the end of the chain is close to pure noise.
    /// </para>
    /// <para>
    /// Default: 0.1, the paper's beta_N (Rasul et al. 2021, Sec. 4).
    /// </para>
    /// </remarks>
    public double BetaEnd { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets how beta grows from BetaStart to BetaEnd.
    /// </summary>
    /// <value>A noise schedule; Linear by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The shape of the noise curve across the steps: linear grows at a constant rate.
    /// </para>
    /// <para>
    /// Default: Linear, the paper's schedule (Rasul et al. 2021, Sec. 4).
    /// </para>
    /// </remarks>
    public AiDotNet.Enums.BetaSchedule BetaSchedule { get; set; } = AiDotNet.Enums.BetaSchedule.Linear;

    /// <summary>
    /// Gets or sets how many sample paths a forecast generates.
    /// </summary>
    /// <value>The number of sampled forecasts; 100 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TimeGrad predicts a distribution. It draws this many possible futures; the point forecast is their mean and quantiles summarize their spread.
    /// </para>
    /// <para>
    /// Default: 100, the number of samples the paper evaluates with (Rasul et al. 2021, Sec. 4).
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets the dropout applied between stacked RNN layers.
    /// </summary>
    /// <value>A probability in [0, 1); 0.1 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Randomly ignores some values during training so the model does not memorize the data.
    /// </para>
    /// <para>
    /// Default: 0.1, the reference implementation's RNN dropout.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the width the diffusion-step embedding is projected to before each residual block.
    /// </summary>
    /// <value>The step-embedding width; 64 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The denoiser is told which step it is on through this embedding; wider lets it distinguish steps more finely.
    /// </para>
    /// <para>
    /// Default: 64, the reference implementation's residual_hidden.
    /// </para>
    /// </remarks>
    public int DenoisingNetworkDim { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of gated residual blocks in the denoiser.
    /// </summary>
    /// <value>The denoiser depth; 8 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many processing blocks the noise predictor stacks.
    /// </para>
    /// <para>
    /// Default: 8, the paper's denoiser depth (Rasul et al. 2021, Sec. 4).
    /// </para>
    /// </remarks>
    public int ResidualLayers { get; set; } = 8;

    /// <summary>
    /// Gets or sets the channel width of each residual block.
    /// </summary>
    /// <value>The denoiser width; 8 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many features each denoiser block works with.
    /// </para>
    /// <para>
    /// Default: 8, the paper's residual channels (Rasul et al. 2021, Sec. 4).
    /// </para>
    /// </remarks>
    public int ResidualChannels { get; set; } = 8;

    /// <summary>
    /// Gets or sets the dilation cycle of the residual blocks: block i dilates by 2^(i mod cycle).
    /// </summary>
    /// <value>The cycle length; 2 by default, giving dilations 1, 2, 1, 2, and so on.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dilation lets a block look at values further apart; the cycle controls how far.
    /// </para>
    /// <para>
    /// Default: 2, the paper's dilation cycle (Rasul et al. 2021, Sec. 4).
    /// </para>
    /// </remarks>
    public int DilationCycleLength { get; set; } = 2;

    /// <summary>
    /// Gets or sets the size E of the sinusoidal diffusion-step embedding, which has 2E entries.
    /// </summary>
    /// <value>E; 16 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The step number is turned into 2E sine and cosine values before the network reads it.
    /// </para>
    /// <para>
    /// Default: 16, the reference implementation's time_emb_dim.
    /// </para>
    /// </remarks>
    public int TimeEmbeddingDim { get; set; } = 16;
}
