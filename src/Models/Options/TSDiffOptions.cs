using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TSDiff (Time Series Diffusion for unconditional/conditional generation).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// TSDiff is a flexible diffusion model for time series that supports both unconditional
/// generation and various conditioning mechanisms for forecasting and imputation.
/// </para>
/// <para><b>For Beginners:</b> TSDiff is designed as a versatile time series generator
/// that can handle multiple tasks with one architecture.
///
/// <b>The Key Insight:</b>
/// Different time series tasks (forecasting, imputation, generation) can all be viewed
/// as conditional generation problems. TSDiff uses a unified framework with different
/// conditioning strategies:
///
/// <b>Supported Tasks:</b>
/// 1. <b>Unconditional Generation:</b> Generate synthetic time series from scratch
/// 2. <b>Forecasting:</b> Condition on historical data to predict future
/// 3. <b>Imputation:</b> Condition on observed values to fill missing
/// 4. <b>Refinement:</b> Condition on noisy data to produce clean version
///
/// <b>TSDiff Architecture:</b>
/// - Self-guided diffusion: Uses attention over time for temporal coherence
/// - Observation guidance: Gradient-based conditioning on observations
/// - Flexible scheduler: Different noise schedules for different tasks
/// - Multi-resolution: Captures patterns at multiple time scales
///
/// <b>Key Benefits:</b>
/// - Single model for multiple tasks
/// - Can combine conditioning strategies
/// - Generates long, coherent sequences
/// - Captures complex temporal dynamics
/// </para>
/// <para>
/// <b>Reference:</b> Kollovieh et al., "Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting", 2023.
/// https://arxiv.org/abs/2307.11494
/// </para>
/// </remarks>
public class TSDiffOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TSDiffOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default TSDiff configuration suitable for
    /// probabilistic time series forecasting with self-guided diffusion.
    /// </para>
    /// </remarks>
    public TSDiffOptions()
    {
        // Seed is INHERITED from ModelOptions and set here rather than shadowed with `new`,
        // which would leave anything holding a ModelOptions reference reading null.
        //
        // Defaulted so repeated predictions agree. This does not collapse the sampling: the
        // NumSamples paths still differ from one another, so the spread stays a real estimate
        // and the intervals the paper reports remain meaningful. It fixes only that Predict
        // called twice on the same input returns the same answer - the sampler equivalent of
        // seeding a generator before inference, not of switching sampling off. Set it to
        // another value for a different sample set, or vary it per call for independent draws.
        Seed = 1;
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TSDiffOptions(TSDiffOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        SequenceLength = other.SequenceLength;
        ForecastHorizon = other.ForecastHorizon;
        NumFeatures = other.NumFeatures;
        HiddenDimension = other.HiddenDimension;
        NumResidualBlocks = other.NumResidualBlocks;
        NumDiffusionSteps = other.NumDiffusionSteps;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        BetaSchedule = other.BetaSchedule;
        NumSamples = other.NumSamples;
        GuidanceScale = other.GuidanceScale;
        UnconditionalProbability = other.UnconditionalProbability;
        TrainingBatchSize = other.TrainingBatchSize;
        LearningRate = other.LearningRate;
        UseSelfGuidance = other.UseSelfGuidance;
        UseObservationGuidance = other.UseObservationGuidance;
        DropoutRate = other.DropoutRate;
        NumAttentionHeads = other.NumAttentionHeads;
        KernelSize = other.KernelSize;
    }

    /// <summary>
    /// Gets or sets the sequence length (context + forecast).
    /// </summary>
    /// <value>The total sequence length, defaulting to 192.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The total length of sequences the model works with.
    /// For forecasting, this is split into context (past) and horizon (future).
    /// </para>
    /// </remarks>
    public int SequenceLength { get; set; } = 192;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 24.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// The context length is SequenceLength - ForecastHorizon.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the number of features (variables).
    /// </summary>
    /// <value>The number of features, defaulting to 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many variables are measured at each time step.
    /// TSDiff can handle multivariate time series.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 1;

    /// <summary>
    /// Gets or sets the hidden dimension for the denoising network.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 64 - the paper's residual width.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size of the network.
    /// Larger values can capture more complex patterns.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of residual blocks in the denoising network.
    /// </summary>
    /// <value>The number of residual blocks, defaulting to 3 - the paper's backbone depth.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Residual blocks are the building blocks of the network.
    /// More blocks = deeper network = more capacity for complex patterns.
    /// </para>
    /// </remarks>
    public int NumResidualBlocks { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of diffusion steps.
    /// </summary>
    /// <value>The number of diffusion steps, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many steps in the noise addition/removal process.
    /// More steps = higher quality but slower.
    /// </para>
    /// </remarks>
    public int NumDiffusionSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the starting noise level (beta_1).
    /// </summary>
    /// <value>The starting beta, defaulting to 0.0001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Initial variance of noise in forward diffusion.
    /// Small values mean gentle noise at first.
    /// </para>
    /// </remarks>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the ending noise level (beta_T).
    /// </summary>
    /// <value>The ending beta, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// Kollovieh et al. 2023 Appendix: "The number of timesteps in the diffusion process was set
    /// to T = 100 and we used a linear scheduler with beta_1 = 0.0001 and beta_100 = 0.1."
    /// </para>
    /// <para>
    /// The value belongs WITH the step count. Ho et al. 2020 pair beta_T = 0.02 with T = 1000,
    /// where the cumulative product alphaBar_T lands near 4e-5 - x_T is then indistinguishable
    /// from pure noise, which is what the sampler starts from. Carrying 0.02 over to T = 100
    /// leaves alphaBar_T near 0.37, so the forward process still retains ~61% of the signal
    /// while the sampler still starts at pure noise: the two ends no longer meet, and no amount
    /// of training closes that gap.
    /// </para>
    /// <para><b>For Beginners:</b> Final variance of noise. By the last step,
    /// data should be approximately standard Gaussian.
    /// </para>
    /// </remarks>
    public double BetaEnd { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the noise schedule type.
    /// </summary>
    /// <value>The beta schedule, defaulting to "linear".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How noise levels change across steps:
    /// - "linear": Uniform increase
    /// - "cosine": Smoother, often better for generation
    /// - "quadratic": Faster increase at the end
    /// </para>
    /// </remarks>
    public string BetaSchedule { get; set; } = "linear";

    /// <summary>
    /// Gets or sets the number of samples for uncertainty estimation.
    /// </summary>
    /// <value>The number of samples, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different forecasts to generate.
    /// More samples = better uncertainty estimates.
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets the guidance scale for classifier-free guidance.
    /// </summary>
    /// <value>The guidance scale, defaulting to 1.0 (no guidance boost).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how strongly conditioning affects generation.
    /// Values &gt; 1 make outputs more consistent with conditions but less diverse.
    ///
    /// <b>Typical values:</b>
    /// - 1.0: Standard generation
    /// - 2.0-4.0: Moderate guidance
    /// - &gt;4.0: Strong guidance (less diversity)
    /// </para>
    /// </remarks>
    public double GuidanceScale { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the probability of dropping the conditioning during a training step.
    /// </summary>
    /// <value>Defaults to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sampling can blend a conditioned forecast with an
    /// unconditional one to sharpen it (see <see cref="GuidanceScale"/>). For that blend to mean
    /// anything, the model has to have been trained to forecast without the conditioning some of
    /// the time. This is how often that happens.</para>
    /// <para><b>Provenance:</b> 0.1 is the conditioning dropout rate Ho and Salimans,
    /// "Classifier-Free Diffusion Guidance" (2022) Section 3 use to train one network jointly on
    /// the conditional and unconditional objectives. Set it to 0 to train a purely conditional
    /// denoiser, in which case leave <see cref="GuidanceScale"/> at 1.0.</para>
    /// </remarks>
    public double UnconditionalProbability { get; set; } = 0.1;

    /// <summary>
    /// Number of (timestep, noise) draws averaged into a single training step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Ho et al. (2020), "Denoising Diffusion Probabilistic Models", Algorithm 1 draws one
    /// timestep per example and averages the step over a minibatch - 128 examples in their
    /// Section 4. A caller here supplies one example at a time, so the averaging has to happen
    /// over the noise process instead: without it, a step's gradient (and its reported loss) is a
    /// one-sample estimate of an expectation taken over every noise level, and successive steps
    /// differ mostly by which timestep came up.
    /// </para>
    /// <para><b>For Beginners:</b> Diffusion training asks "given this partly noised series, what
    /// noise was added?" at a randomly chosen noise level. Asking once gives a very jumpy answer;
    /// asking 32 times at different levels and averaging gives a steady one. Raise this for
    /// smoother training at proportionally more work per step, lower it to train faster.</para>
    /// </remarks>
    public int TrainingBatchSize { get; set; } = 32;

    /// <summary>
    /// Learning rate for the default Adam optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Kollovieh et al. (2023), "Predict, Refine, Synthesize: Self-Guiding Diffusion Models for
    /// Probabilistic Time Series Forecasting", train TSDiff with Adam at 1e-3, the same rate
    /// Tashiro et al. (2021) use for CSDI. Without this the model fell back to
    /// <see cref="OptimizationAlgorithmOptions{T, TInput, TOutput}.InitialLearningRate"/>'s
    /// generic 0.01, which is 10x that rate and 50x the 2e-4 of Ho et al. (2020) Section 4 - at
    /// that step size the epsilon-prediction loss rises instead of falling.
    /// </para>
    /// <para><b>For Beginners:</b> This is how big a step training takes each time it learns
    /// something. Too big and the model overshoots and gets worse; too small and it barely moves.
    /// 0.001 is the value the papers in this family use.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets whether to use self-guidance during sampling.
    /// </summary>
    /// <value>True to use self-guidance; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Self-guidance uses the model's own predictions
    /// to refine generation, improving temporal coherence. The model essentially
    /// "checks its own work" during generation.
    /// </para>
    /// </remarks>
    public bool UseSelfGuidance { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use observation guidance for conditioning.
    /// </summary>
    /// <value>True to use observation guidance; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Observation guidance uses gradient-based steering
    /// to ensure generated values match observed/conditioned values. Essential for
    /// forecasting and imputation tasks.
    /// </para>
    /// </remarks>
    public bool UseObservationGuidance { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout randomly disables neurons during training
    /// to prevent overfitting.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of attention heads in self-attention layers.
    /// </summary>
    /// <value>The number of attention heads, defaulting to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Attention heads allow the model to attend to
    /// different aspects of the time series simultaneously.
    /// </para>
    /// </remarks>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the kernel size for temporal convolutions.
    /// </summary>
    /// <value>The kernel size, defaulting to 3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the sliding window for local patterns.
    /// Larger kernels capture longer local dependencies but need more computation.
    /// </para>
    /// </remarks>
    public int KernelSize { get; set; } = 3;
}
