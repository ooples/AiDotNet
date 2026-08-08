namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Bayesian Network Synthesis, a statistical approach that
/// learns a directed acyclic graph (DAG) structure and conditional probability tables
/// to generate synthetic tabular data via ancestral sampling.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Bayesian Network Synthesis operates in three phases:
/// - <b>Structure learning</b>: Discovers a DAG using greedy hill-climbing with BIC scoring
/// - <b>Parameter estimation</b>: Estimates conditional probability tables (CPTs) from the data
/// - <b>Ancestral sampling</b>: Generates data by sampling from root nodes down through the DAG
/// </para>
/// <para>
/// <b>For Beginners:</b> This method creates a probabilistic model of your data:
///
/// Think of a family tree of features — some features "depend on" others.
/// For example, in a health dataset:
/// 1. Age has no parents (sampled first)
/// 2. Blood pressure depends on Age
/// 3. Medication depends on Blood pressure
///
/// The model learns these dependency chains and samples new data following
/// the same parent-to-child order, producing statistically coherent rows.
///
/// Unlike neural network generators (CTGAN, TVAE), this uses classical statistics,
/// making it faster to train and more interpretable, though less flexible for
/// complex distributions.
///
/// Example:
/// <code>
/// var options = new BayesianNetworkSynthOptions&lt;double&gt;
/// {
///     MaxParents = 3,
///     NumBins = 20
/// };
/// var bnSynth = new BayesianNetworkSynthGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// </remarks>
public class BayesianNetworkSynthOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the maximum number of parents per node in the DAG.
    /// </summary>
    /// <value>Maximum parents, defaulting to 3. Higher values allow more complex dependencies but increase computation.</value>
    public int MaxParents { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of discretization bins for continuous features.
    /// </summary>
    /// <value>Number of bins, defaulting to 20.</value>
    public int NumBins { get; set; } = 20;

    /// <summary>
    /// Gets or sets the maximum number of structure learning iterations.
    /// </summary>
    /// <value>Maximum iterations, defaulting to 100.</value>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the Laplace smoothing constant for CPT estimation.
    /// </summary>
    /// <value>Smoothing constant, defaulting to 1.0. Prevents zero-probability entries in CPTs.</value>
    /// <remarks>
    /// This is a SMOOTHING PRIOR, not a privacy mechanism — it has nothing to do with the Laplace
    /// NOISE that provides differential privacy (see <see cref="PrivacyBudget"/>). The two are easy to
    /// confuse because both are named after the same distribution.
    /// </remarks>
    public double LaplaceSmoothing { get; set; } = 1.0;

    #region Differential Privacy (PrivBayes)

    /// <summary>
    /// Gets or sets whether to enforce differential privacy, which is what makes this PrivBayes
    /// rather than a plain Bayesian-network synthesizer.
    /// </summary>
    /// <value>
    /// Defaults to <c>true</c>. PrivBayes exists to release data privately; running without the
    /// privacy mechanisms gives a generator that offers NO privacy guarantee whatsoever, so that has
    /// to be an explicit opt-out rather than the default.
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> differential privacy is a mathematical guarantee that the released data
    /// cannot reveal whether any single individual was in the original dataset. It is achieved by
    /// injecting a carefully calibrated amount of random noise. Turning this off makes generation
    /// more faithful to the input data but removes the guarantee entirely.
    /// </para>
    /// </remarks>
    public bool EnableDifferentialPrivacy { get; set; } = true;

    /// <summary>
    /// Gets or sets the total privacy budget, epsilon.
    /// </summary>
    /// <value>
    /// Defaults to 1.0 — the least-private setting in the range the paper evaluates
    /// (epsilon in {0.05, 0.1, 0.2, 0.5, 0.8, 1.0}), chosen so out-of-the-box utility is reasonable
    /// while still providing a real guarantee.
    /// </value>
    /// <remarks>
    /// Smaller epsilon means MORE privacy and more noise. The budget is split between learning the
    /// network structure and adding noise to the marginals — see
    /// <see cref="StructureBudgetFraction"/>.
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when set to a value that is not positive and finite. Epsilon appears in a denominator
    /// when the noise scale is computed, so zero or negative is not a weaker guarantee -- it is no
    /// guarantee, arrived at silently.
    /// </exception>
    public double PrivacyBudget
    {
        get => _privacyBudget;
        set
        {
            if (value <= 0.0 || double.IsNaN(value) || double.IsInfinity(value))
            {
                throw new ArgumentOutOfRangeException(nameof(PrivacyBudget), value,
                    "PrivacyBudget (epsilon) must be a positive, finite number. It divides the noise "
                    + "scale, so a non-positive budget disables the privacy noise rather than "
                    + "tightening it.");
            }

            _privacyBudget = value;
        }
    }

    private double _privacyBudget = 1.0;

    /// <summary>
    /// Gets or sets the fraction of the total privacy budget spent on learning the network structure,
    /// with the remainder spent on noising the conditional distributions.
    /// </summary>
    /// <value>Defaults to 0.5, the paper's even epsilon/2 split between its two phases.</value>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when set outside the exclusive range (0, 1).
    /// </exception>
    /// <remarks>
    /// VALIDATED BECAUSE BOTH ENDPOINTS FAIL SILENTLY, AND ONE OF THEM FAILS UNSAFELY. The second
    /// phase spends <c>PrivacyBudget * (1 - StructureBudgetFraction)</c>, and the noise scale is
    /// computed only when that share is positive -- otherwise it is left at zero. A fraction of 1.0
    /// therefore emits the conditional distributions with NO noise at all while the object still
    /// reports differential privacy as enabled, which is a privacy failure that looks like a working
    /// configuration. A fraction of 0.0 spends nothing on structure and learns it non-privately.
    /// </remarks>
    public double StructureBudgetFraction
    {
        get => _structureBudgetFraction;
        set
        {
            if (value <= 0.0 || value >= 1.0 || double.IsNaN(value))
            {
                throw new ArgumentOutOfRangeException(nameof(StructureBudgetFraction), value,
                    "StructureBudgetFraction must be strictly between 0 and 1. At 1 the second phase "
                    + "receives no budget and its distributions are published without privacy noise; "
                    + "at 0 the structure is learned without any.");
            }

            _structureBudgetFraction = value;
        }
    }

    private double _structureBudgetFraction = 0.5;

    #endregion
}
