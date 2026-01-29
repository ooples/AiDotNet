using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for S4 (Structured State Space Sequence Model).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// S4 is a foundational state space model that achieves near-linear complexity through
/// structured parameterization of the state transition matrix using the HiPPO framework.
/// </para>
/// <para><b>For Beginners:</b> S4 is a breakthrough model that showed state space models
/// can match transformers on long-range sequence tasks:
///
/// <b>The Key Insight:</b>
/// State space models have linear complexity O(n), but historically couldn't compete with
/// attention. S4 solves this by:
/// 1. Using HiPPO (High-order Polynomial Projection Operators) for state initialization
/// 2. Structuring the state matrix A as a diagonal plus low-rank (DPLR) matrix
/// 3. Computing convolutions efficiently using FFT
///
/// <b>How It Works:</b>
/// 1. <b>HiPPO Matrix:</b> Initializes A to optimally compress history into the state
/// 2. <b>DPLR Decomposition:</b> A = diagonal + low-rank for efficient computation
/// 3. <b>Discretization:</b> Converts continuous SSM to discrete for sequence data
/// 4. <b>FFT Convolution:</b> Computes SSM as convolution using O(n log n) FFT
///
/// <b>The Math (simplified):</b>
/// - State update: x' = Ax + Bu (A is HiPPO-structured)
/// - Output: y = Cx + Du
/// - Discretized: x_k = A_bar * x_{k-1} + B_bar * u_k
/// - For long sequences: compute as convolution K * u using FFT
///
/// <b>Advantages:</b>
/// - Near-linear complexity O(n log n) via FFT
/// - Excellent on Long Range Arena benchmark
/// - Handles sequences up to 16K+ tokens
/// - Foundation for Mamba and other modern SSMs
/// </para>
/// <para>
/// <b>Reference:</b> Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces", 2022.
/// https://arxiv.org/abs/2111.00396
/// </para>
/// </remarks>
public class S4Options<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="S4Options{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default S4 configuration optimized for
    /// long-range sequence modeling with near-linear complexity.
    /// </para>
    /// </remarks>
    public S4Options()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public S4Options(S4Options<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        ModelDimension = other.ModelDimension;
        StateDimension = other.StateDimension;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        UseBidirectional = other.UseBidirectional;
        HippoMethod = other.HippoMethod;
        DiscretizationMethod = other.DiscretizationMethod;
        UseLowRankCorrection = other.UseLowRankCorrection;
        LowRankRank = other.LowRankRank;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>The context length, defaulting to 1024.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at.
    /// S4 excels at very long contexts (4K-16K+) due to near-linear complexity.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the model dimension (d_model).
    /// </summary>
    /// <value>The model dimension, defaulting to 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The main hidden dimension of the model.
    /// Controls the capacity for learning patterns.
    /// </para>
    /// </remarks>
    public int ModelDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the state dimension (N in the paper).
    /// </summary>
    /// <value>The state dimension, defaulting to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The dimension of the hidden state in the SSM.
    /// The HiPPO matrix A is N x N. Larger N captures more history but uses more computation.
    /// Typical values: 32, 64, or 128.
    /// </para>
    /// </remarks>
    public int StateDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of S4 layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 6.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many S4 blocks are stacked.
    /// More layers = deeper model = more capacity for complex patterns.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting by randomly
    /// setting some values to zero during training.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use bidirectional processing.
    /// </summary>
    /// <value>True for bidirectional; false for causal/unidirectional. Default: false.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Bidirectional S4 processes the sequence both
    /// forward and backward, which can improve accuracy on classification tasks.
    /// Use unidirectional (false) for autoregressive tasks like forecasting.
    /// </para>
    /// </remarks>
    public bool UseBidirectional { get; set; } = false;

    /// <summary>
    /// Gets or sets the HiPPO method for state matrix initialization.
    /// </summary>
    /// <value>The HiPPO method, defaulting to "legs" (Legendre).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> HiPPO defines how the state matrix A is initialized:
    /// - "legs": Legendre polynomials - good general purpose (recommended)
    /// - "legt": Translated Legendre - for fixed-length context
    /// - "lagt": Laguerre - for infinite context with exponential decay
    /// - "fourier": Fourier basis - for periodic signals
    /// </para>
    /// </remarks>
    public string HippoMethod { get; set; } = "legs";

    /// <summary>
    /// Gets or sets the discretization method for converting continuous to discrete SSM.
    /// </summary>
    /// <value>The discretization method, defaulting to "bilinear".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> S4 is defined in continuous time but processes
    /// discrete sequences. This controls how to convert:
    /// - "bilinear": Tustin/trapezoidal method (most accurate, default)
    /// - "zoh": Zero-order hold (faster but less accurate)
    /// </para>
    /// </remarks>
    public string DiscretizationMethod { get; set; } = "bilinear";

    /// <summary>
    /// Gets or sets whether to use low-rank correction in the DPLR decomposition.
    /// </summary>
    /// <value>True to use low-rank correction; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> S4 represents A = diagonal + low-rank (DPLR).
    /// The low-rank correction helps capture the off-diagonal structure of HiPPO.
    /// Disabling this makes computation faster but may reduce accuracy.
    /// </para>
    /// </remarks>
    public bool UseLowRankCorrection { get; set; } = true;

    /// <summary>
    /// Gets or sets the rank of the low-rank correction.
    /// </summary>
    /// <value>The low-rank rank, defaulting to 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> In the DPLR decomposition A = Lambda + P*Q^T,
    /// this is the rank of P*Q^T. Rank 1 is sufficient for most HiPPO matrices.
    /// Higher rank provides more expressiveness at the cost of computation.
    /// </para>
    /// </remarks>
    public int LowRankRank { get; set; } = 1;
}
