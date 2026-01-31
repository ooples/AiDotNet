using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for CSDI (Conditional Score-based Diffusion model for Imputation).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// CSDI is a probabilistic model for time series imputation that uses score-based
/// diffusion to fill in missing values with well-calibrated uncertainty estimates.
/// </para>
/// <para><b>For Beginners:</b> CSDI solves a critical problem in real-world data:
/// missing values. Instead of simple interpolation, it generates plausible values
/// that are consistent with the observed data.
///
/// <b>The Key Insight:</b>
/// Unlike TimeGrad which forecasts future values, CSDI focuses on imputation:
/// filling in missing values WITHIN the observed time series. It conditions on
/// what you DO know to infer what you DON'T know.
///
/// <b>How CSDI Works:</b>
/// 1. <b>Conditional Masking:</b> Identify which values are observed vs missing
/// 2. <b>Score Matching:</b> Learn the gradient of log probability (the "score")
/// 3. <b>Reverse Diffusion:</b> Start from noise, gradually denoise conditioned on observed values
/// 4. <b>Imputation:</b> Generate multiple samples for uncertainty quantification
///
/// <b>Architecture:</b>
/// - Transformer-based score network with self-attention
/// - Temporal and feature embeddings for position encoding
/// - Conditional U-Net style residual blocks
/// - Side information integration for covariates
///
/// <b>Key Benefits:</b>
/// - Handles arbitrary missing patterns (not just regular gaps)
/// - Provides uncertainty estimates for imputed values
/// - Can incorporate side information (covariates)
/// - State-of-the-art imputation quality
/// </para>
/// <para>
/// <b>Reference:</b> Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", 2021.
/// https://arxiv.org/abs/2107.03502
/// </para>
/// </remarks>
public class CSDIOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CSDIOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default CSDI configuration optimized for
    /// probabilistic time series imputation with score-based diffusion.
    /// </para>
    /// </remarks>
    public CSDIOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CSDIOptions(CSDIOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SequenceLength = other.SequenceLength;
        NumFeatures = other.NumFeatures;
        HiddenDimension = other.HiddenDimension;
        NumResidualLayers = other.NumResidualLayers;
        NumDiffusionSteps = other.NumDiffusionSteps;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        BetaSchedule = other.BetaSchedule;
        NumSamples = other.NumSamples;
        NumHeads = other.NumHeads;
        TimeEmbeddingDim = other.TimeEmbeddingDim;
        FeatureEmbeddingDim = other.FeatureEmbeddingDim;
        DropoutRate = other.DropoutRate;
        UseSideInfo = other.UseSideInfo;
        SideInfoDim = other.SideInfoDim;
        UseAttention = other.UseAttention;
    }

    /// <summary>
    /// Gets or sets the sequence length (time steps).
    /// </summary>
    /// <value>The sequence length, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many time steps are in each sequence.
    /// This includes both observed and potentially missing values.
    /// </para>
    /// </remarks>
    public int SequenceLength { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of features (variables).
    /// </summary>
    /// <value>The number of features, defaulting to 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many variables are measured at each time step.
    /// CSDI can impute multivariate time series with complex dependencies.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 1;

    /// <summary>
    /// Gets or sets the hidden dimension for the score network.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size of the neural network.
    /// Larger values capture more complex patterns but require more computation.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of residual layers in the score network.
    /// </summary>
    /// <value>The number of residual layers, defaulting to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many residual blocks process the data.
    /// Residual connections help gradients flow and enable deeper networks.
    /// </para>
    /// </remarks>
    public int NumResidualLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of diffusion steps.
    /// </summary>
    /// <value>The number of diffusion steps, defaulting to 50.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many steps in the diffusion process.
    /// More steps = better quality but slower. CSDI typically uses fewer
    /// steps than image diffusion models because time series are lower-dimensional.
    /// </para>
    /// </remarks>
    public int NumDiffusionSteps { get; set; } = 50;

    /// <summary>
    /// Gets or sets the starting noise level (beta_1).
    /// </summary>
    /// <value>The starting beta, defaulting to 0.0001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Initial variance of noise added in forward diffusion.
    /// Small values mean the first step barely perturbs the data.
    /// </para>
    /// </remarks>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the ending noise level (beta_T).
    /// </summary>
    /// <value>The ending beta, defaulting to 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Final variance of noise. By step T, the data
    /// should be approximately standard Gaussian noise.
    /// Note: CSDI uses a higher beta_end than TimeGrad because imputation
    /// requires more aggressive noise schedules.
    /// </para>
    /// </remarks>
    public double BetaEnd { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the noise schedule type.
    /// </summary>
    /// <value>The beta schedule, defaulting to "quad".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How noise levels change across diffusion steps:
    /// - "quad": Quadratic schedule (default for CSDI, smooth transition)
    /// - "linear": Uniform increase from beta_start to beta_end
    /// - "cosine": Smoother schedule based on cosine function
    /// </para>
    /// </remarks>
    public string BetaSchedule { get; set; } = "quad";

    /// <summary>
    /// Gets or sets the number of samples to generate for uncertainty estimation.
    /// </summary>
    /// <value>The number of samples, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different imputation paths to generate.
    /// More samples = better uncertainty estimates but slower inference.
    /// The variance across samples indicates imputation uncertainty.
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of attention heads in the transformer layers.
    /// </summary>
    /// <value>The number of attention heads, defaulting to 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Attention heads let the model attend to
    /// different aspects of the input simultaneously. Multiple heads capture
    /// different types of temporal and feature dependencies.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the dimension of time step embeddings.
    /// </summary>
    /// <value>The time embedding dimension, defaulting to 128.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each diffusion time step gets an embedding vector.
    /// This helps the network understand "how much noise" is present at each step.
    /// Uses sinusoidal embeddings similar to positional encodings in transformers.
    /// </para>
    /// </remarks>
    public int TimeEmbeddingDim { get; set; } = 128;

    /// <summary>
    /// Gets or sets the dimension of feature embeddings.
    /// </summary>
    /// <value>The feature embedding dimension, defaulting to 16.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each feature (variable) gets a learned embedding.
    /// This helps the model understand which feature is which, similar to
    /// how word embeddings represent words in NLP.
    /// </para>
    /// </remarks>
    public int FeatureEmbeddingDim { get; set; } = 16;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout randomly disables neurons during training
    /// to prevent overfitting. Applied in attention and feedforward layers.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use side information (covariates).
    /// </summary>
    /// <value>True to use side information; false otherwise. Default: false.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Side information includes additional variables
    /// that help predict missing values (e.g., day of week, weather conditions).
    /// When enabled, the model conditions on these auxiliary features.
    /// </para>
    /// </remarks>
    public bool UseSideInfo { get; set; } = false;

    /// <summary>
    /// Gets or sets the dimension of side information features.
    /// </summary>
    /// <value>The side information dimension, defaulting to 0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If using side information, this specifies
    /// how many additional features are provided as context for imputation.
    /// </para>
    /// </remarks>
    public int SideInfoDim { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to use self-attention in the score network.
    /// </summary>
    /// <value>True to use attention; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Self-attention allows the model to consider
    /// relationships between ALL positions when imputing each value.
    /// This is crucial for capturing long-range dependencies in time series.
    /// </para>
    /// </remarks>
    public bool UseAttention { get; set; } = true;
}
