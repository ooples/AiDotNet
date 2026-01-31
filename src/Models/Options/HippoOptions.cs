using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for HiPPO (High-order Polynomial Projection Operators) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// HiPPO provides the theoretical foundation for efficient state space models like S4 and Mamba.
/// It defines optimal state matrices for compressing sequential input history into a fixed-size state.
/// </para>
/// <para><b>For Beginners:</b> HiPPO answers a fundamental question in sequence modeling:
/// "How do we optimally remember a continuous history in a fixed-size memory?"
///
/// <b>The Key Insight:</b>
/// When processing a sequence, we want our hidden state to be an "optimal summary" of the past.
/// HiPPO shows that by projecting the input history onto polynomial bases (like Legendre polynomials),
/// we can create hidden states that provably capture the history optimally.
///
/// <b>How It Works:</b>
/// 1. <b>Polynomial Basis:</b> Choose a basis (Legendre, Laguerre, Fourier, etc.)
/// 2. <b>Optimal Projection:</b> The state x(t) represents coefficients of the polynomial
///    approximation of the input history
/// 3. <b>Online Update:</b> The state can be updated efficiently as new inputs arrive
/// 4. <b>Memory Matrix A:</b> Defines how the state evolves (different for each basis)
///
/// <b>The Math (simplified):</b>
/// State Space Model: dx/dt = Ax + Bu
/// - A is the "HiPPO matrix" derived from the chosen polynomial basis
/// - x(t) contains coefficients: history ≈ Σ x_i(t) * P_i(τ)
/// - Different A matrices give different memory properties:
///   - LegS: Sliding window over recent history
///   - LegT: Fixed window over [0, t]
///   - LagT: Exponential decay (older = less weight)
///
/// <b>Why HiPPO Matters:</b>
/// - Provides principled initialization for state space models
/// - Enables models to handle very long sequences
/// - Foundation for S4, Mamba, and other modern SSMs
/// - Mathematically optimal for memory compression
/// </para>
/// <para>
/// <b>Reference:</b> Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections", 2020.
/// https://arxiv.org/abs/2008.07669
/// </para>
/// </remarks>
public class HippoOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="HippoOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default HiPPO configuration optimized for
    /// sequence modeling with principled memory compression.
    /// </para>
    /// </remarks>
    public HippoOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public HippoOptions(HippoOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        ModelDimension = other.ModelDimension;
        StateDimension = other.StateDimension;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        HippoMethod = other.HippoMethod;
        DiscretizationMethod = other.DiscretizationMethod;
        TimescaleMin = other.TimescaleMin;
        TimescaleMax = other.TimescaleMax;
        UseNormalization = other.UseNormalization;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>The context length, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at.
    /// HiPPO's polynomial projection allows efficient handling of long sequences.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

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
    /// <para><b>For Beginners:</b> The dimension of the HiPPO state (polynomial order).
    /// The HiPPO matrix A is N x N. Larger N = more accurate history approximation
    /// but more computation. Typical values: 32, 64, 128.
    ///
    /// <b>Analogy:</b> Like choosing how many terms to keep in a Fourier series.
    /// More terms = more accurate, but more work to compute.
    /// </para>
    /// </remarks>
    public int StateDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of HiPPO layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many HiPPO blocks are stacked.
    /// More layers = deeper model = more capacity for complex patterns.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the HiPPO method for state matrix initialization.
    /// </summary>
    /// <value>The HiPPO method, defaulting to "legs" (Legendre Scaled).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> HiPPO defines several polynomial bases, each with
    /// different memory properties:
    ///
    /// <b>Available Methods:</b>
    /// - "legs" (Legendre Scaled): Sliding window over recent history
    ///   - Best for: General purpose, variable-length sequences
    ///   - Memory: Uniform attention over a sliding window
    ///
    /// - "legt" (Legendre Translated): Fixed window [0, t]
    ///   - Best for: Fixed-length context
    ///   - Memory: Uniform attention over entire history
    ///
    /// - "lagt" (Laguerre Translated): Exponential decay
    ///   - Best for: Infinite horizon with decaying importance
    ///   - Memory: Recent events weighted more than distant
    ///
    /// - "fourier": Fourier basis
    ///   - Best for: Periodic/cyclical data
    ///   - Memory: Captures frequency components
    /// </para>
    /// </remarks>
    public string HippoMethod { get; set; } = "legs";

    /// <summary>
    /// Gets or sets the discretization method for converting continuous to discrete SSM.
    /// </summary>
    /// <value>The discretization method, defaulting to "bilinear".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> HiPPO is defined in continuous time but we process
    /// discrete sequences. This controls the conversion:
    /// - "bilinear": Tustin/trapezoidal method (most accurate, preserves stability)
    /// - "zoh": Zero-order hold (faster but less accurate)
    /// - "euler": Euler method (simplest but can be unstable)
    /// </para>
    /// </remarks>
    public string DiscretizationMethod { get; set; } = "bilinear";

    /// <summary>
    /// Gets or sets the minimum timescale for the SSM.
    /// </summary>
    /// <value>The minimum timescale, defaulting to 0.001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls the finest temporal resolution the model
    /// can represent. Smaller values capture faster dynamics but may be unstable.
    /// This is the Δt_min parameter in discretization.
    /// </para>
    /// </remarks>
    public double TimescaleMin { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the maximum timescale for the SSM.
    /// </summary>
    /// <value>The maximum timescale, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls the longest temporal dependency the model
    /// can capture efficiently. Larger values capture slower trends but may be
    /// less sensitive to fast changes. This is the Δt_max parameter.
    /// </para>
    /// </remarks>
    public double TimescaleMax { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use normalization between HiPPO layers.
    /// </summary>
    /// <value>True to use layer normalization; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Layer normalization helps stabilize training
    /// by keeping activations in a reasonable range. Generally recommended.
    /// </para>
    /// </remarks>
    public bool UseNormalization { get; set; } = true;
}
