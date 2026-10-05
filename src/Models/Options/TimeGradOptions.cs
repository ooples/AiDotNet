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

    /// <summary>Gets or sets the number of past steps the model conditions on.</summary>
    public int ContextLength { get; set; } = 168;

    /// <summary>Gets or sets the number of future steps to forecast.</summary>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>Gets or sets the hidden size of each RNN layer (40 in the paper).</summary>
    public int HiddenDimension { get; set; } = 40;

    /// <summary>Gets or sets the number of stacked LSTM layers in the history encoder (2 in the paper).</summary>
    public int NumRnnLayers { get; set; } = 2;

    /// <summary>Gets or sets the number of diffusion steps N (100 in the paper).</summary>
    public int NumDiffusionSteps { get; set; } = 100;

    /// <summary>Gets or sets the first noise variance beta_1 (1e-4 in the paper).</summary>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>Gets or sets the last noise variance beta_N (0.1 in the paper).</summary>
    public double BetaEnd { get; set; } = 0.1;

    /// <summary>Gets or sets how beta grows from <see cref="BetaStart"/> to <see cref="BetaEnd"/> (linear in the paper).</summary>
    public AiDotNet.Enums.BetaSchedule BetaSchedule { get; set; } = AiDotNet.Enums.BetaSchedule.Linear;

    /// <summary>Gets or sets how many sample paths a forecast averages (and quantile forecasts summarize).</summary>
    public int NumSamples { get; set; } = 100;

    /// <summary>Gets or sets the dropout applied between stacked RNN layers (0.1 in the reference).</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the width the diffusion-step embedding is projected to before each residual block
    /// (<c>residual_hidden</c> in the reference implementation, 64).
    /// </summary>
    public int DenoisingNetworkDim { get; set; } = 64;

    /// <summary>Gets or sets the number of gated residual blocks in the denoiser (8 in the paper).</summary>
    public int ResidualLayers { get; set; } = 8;

    /// <summary>Gets or sets the channel width of each residual block (8 in the paper).</summary>
    public int ResidualChannels { get; set; } = 8;

    /// <summary>
    /// Gets or sets the dilation cycle: block i dilates by 2^(i mod cycle) (2 in the paper, so 1, 2, 1, 2, ...).
    /// </summary>
    public int DilationCycleLength { get; set; } = 2;

    /// <summary>Gets or sets the size E of the sinusoidal diffusion-step embedding, which has 2E entries (16 in the reference).</summary>
    public int TimeEmbeddingDim { get; set; } = 16;
}