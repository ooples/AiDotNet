using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for ScoreGrad (Score-based Gradient Models for Time Series).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// ScoreGrad is a score-based generative model for time series that learns the gradient
/// of the log probability density (score function) for denoising and generation.
/// </para>
/// <para><b>For Beginners:</b> ScoreGrad uses a different approach to diffusion models
/// called "score matching":
///
/// <b>The Key Insight:</b>
/// Instead of learning to predict noise directly, ScoreGrad learns the "score" -
/// the direction that points toward higher probability regions. Following the score
/// gradient leads the model from noise toward realistic time series.
///
/// <b>What is the Score Function?</b>
/// The score is the gradient of the log probability: ∇_x log p(x).
/// - It points toward regions of high probability
/// - Following it uphill finds the most likely data
/// - Denoising Score Matching (DSM) provides a way to learn it
///
/// <b>How ScoreGrad Works:</b>
/// 1. <b>Score Network:</b> Train a network to predict ∇_x log p(x|σ) for various noise levels
/// 2. <b>Noise Conditioning:</b> Condition on noise level σ for multi-scale scores
/// 3. <b>Langevin Dynamics:</b> Use stochastic gradient ascent to sample from the distribution
/// 4. <b>Annealed Sampling:</b> Start with high noise, gradually reduce for refinement
///
/// <b>ScoreGrad Architecture:</b>
/// - Score Network: Predicts gradient direction at each noise level
/// - Noise Embedding: Encodes current noise level for conditioning
/// - Time Embedding: Encodes temporal position information
/// - Skip Connections: Preserves input details during score computation
///
/// <b>Key Benefits:</b>
/// - Principled probabilistic foundation (score matching)
/// - Flexible noise schedules
/// - Can use Langevin dynamics for sampling
/// - Works well for time series with complex dynamics
/// </para>
/// <para>
/// <b>Reference:</b> Yan et al., "ScoreGrad: Multivariate Probabilistic Time Series Forecasting with Continuous Energy-based Generative Models", 2021.
/// https://arxiv.org/abs/2106.10121
/// </para>
/// </remarks>
public class ScoreGradOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ScoreGradOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default ScoreGrad configuration optimized for
    /// probabilistic time series forecasting using score-based generative modeling.
    /// </para>
    /// </remarks>
    public ScoreGradOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ScoreGradOptions(ScoreGradOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SequenceLength = other.SequenceLength;
        ForecastHorizon = other.ForecastHorizon;
        NumFeatures = other.NumFeatures;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumNoiseScales = other.NumNoiseScales;
        SigmaMin = other.SigmaMin;
        SigmaMax = other.SigmaMax;
        NumLangevinSteps = other.NumLangevinSteps;
        StepSize = other.StepSize;
        NumSamples = other.NumSamples;
        DropoutRate = other.DropoutRate;
        UseAnnealing = other.UseAnnealing;
        AnnealingPower = other.AnnealingPower;
    }

    /// <summary>
    /// Gets or sets the sequence length (context length).
    /// </summary>
    /// <value>The sequence length, defaulting to 168.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model uses as context
    /// for computing the score function. Longer sequences capture more history.
    /// </para>
    /// </remarks>
    public int SequenceLength { get; set; } = 168;

    /// <summary>
    /// Gets or sets the forecast horizon.
    /// </summary>
    /// <value>The forecast horizon, defaulting to 24.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// The score network learns to generate this many future time steps.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the number of features.
    /// </summary>
    /// <value>The number of features, defaulting to 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many variables are measured at each time step.
    /// ScoreGrad handles multivariate time series naturally.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 1;

    /// <summary>
    /// Gets or sets the hidden dimension of the score network.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 128.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size of the score network.
    /// Larger dimensions can capture more complex score functions.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of layers in the score network.
    /// </summary>
    /// <value>The number of layers, defaulting to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Depth of the network that predicts scores.
    /// More layers allow learning more complex score patterns.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of noise scales (sigma levels).
    /// </summary>
    /// <value>The number of noise scales, defaulting to 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> ScoreGrad learns scores at multiple noise levels.
    /// More levels give smoother transition from noise to data.
    /// The levels are geometric: σ_1, σ_1/r, σ_1/r², ... where r is the ratio.
    /// </para>
    /// </remarks>
    public int NumNoiseScales { get; set; } = 10;

    /// <summary>
    /// Gets or sets the minimum noise level (sigma).
    /// </summary>
    /// <value>The minimum sigma, defaulting to 0.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The smallest noise level during sampling.
    /// Smaller values give cleaner outputs but may affect diversity.
    /// </para>
    /// </remarks>
    public double SigmaMin { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum noise level (sigma).
    /// </summary>
    /// <value>The maximum sigma, defaulting to 1.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The largest noise level. Data is nearly destroyed
    /// at this level, providing a good starting point for sampling.
    /// </para>
    /// </remarks>
    public double SigmaMax { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of Langevin dynamics steps per noise level.
    /// </summary>
    /// <value>The number of Langevin steps, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Langevin dynamics uses the score to sample.
    /// More steps at each noise level gives better samples but is slower.
    /// Each step follows the score gradient plus some random noise.
    /// </para>
    /// </remarks>
    public int NumLangevinSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the step size (epsilon) for Langevin dynamics.
    /// </summary>
    /// <value>The step size, defaulting to 0.00002.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big each Langevin step is.
    /// Smaller steps are more accurate but require more iterations.
    /// The step size is automatically scaled by sigma² during annealing.
    /// </para>
    /// </remarks>
    public double StepSize { get; set; } = 0.00002;

    /// <summary>
    /// Gets or sets the number of samples for uncertainty estimation.
    /// </summary>
    /// <value>The number of samples, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different forecasts to generate.
    /// Each sample follows a different Langevin trajectory for diversity.
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Regularization to prevent overfitting
    /// the score network to training data.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use annealed Langevin sampling.
    /// </summary>
    /// <value>True to use annealing; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Annealed sampling starts at high noise and
    /// gradually reduces it. This is more effective than using a single noise level.
    /// Without annealing, samples may not converge to the data distribution.
    /// </para>
    /// </remarks>
    public bool UseAnnealing { get; set; } = true;

    /// <summary>
    /// Gets or sets the annealing power for noise schedule.
    /// </summary>
    /// <value>The annealing power, defaulting to 2.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how fast noise decreases during annealing.
    /// Higher values spend more time at lower noise levels (more refinement).
    /// - 1.0: Linear decrease
    /// - 2.0: Quadratic (default, more time at lower noise)
    /// - 3.0+: Even more refinement at low noise
    /// </para>
    /// </remarks>
    public double AnnealingPower { get; set; } = 2.0;
}
