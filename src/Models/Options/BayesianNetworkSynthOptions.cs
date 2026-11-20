namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Bayesian Network Synthesis, a statistical approach that
/// learns a directed acyclic graph (DAG) structure and conditional probability tables
/// to generate synthetic tabular data via ancestral sampling.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// With differential privacy enabled (the default) this is PrivBayes (Zhang et al. 2017):
/// - <b>Structure learning</b>: GreedyBayes picks each attribute's parents with the exponential mechanism
/// - <b>Parameter estimation</b>: d - k joint distributions get Laplace noise; conditionals derive from them
/// - <b>Ancestral sampling</b>: Generates data by sampling from root nodes down through the DAG
/// With it disabled, the structure is learned by greedy BIC hill-climbing and the conditional tables are
/// Laplace-smoothed maximum-likelihood estimates (no privacy guarantee).
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
    /// <summary>Initializes a new instance with the default settings.</summary>
    public BayesianNetworkSynthOptions()
    {
    }

    /// <summary>Initializes a new instance by copying every setting of another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
    /// <remarks>The domain bounds are copied into new vectors, so the copy does not share them.</remarks>
    public BayesianNetworkSynthOptions(BayesianNetworkSynthOptions<T> other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        MaxParents = other.MaxParents;
        UsefulnessThreshold = other.UsefulnessThreshold;
        DomainLowerBounds = other.DomainLowerBounds is { } lower ? new Vector<T>(lower.ToArray()) : null;
        DomainUpperBounds = other.DomainUpperBounds is { } upper ? new Vector<T>(upper.ToArray()) : null;
        NumBins = other.NumBins;
        MaxIterations = other.MaxIterations;
        LaplaceSmoothing = other.LaplaceSmoothing;
        EnableDifferentialPrivacy = other.EnableDifferentialPrivacy;
        _privacyBudget = other._privacyBudget;
        _structureBudgetFraction = other._structureBudgetFraction;
    }

    /// <summary>
    /// Gets or sets the maximum number of parents per node in the DAG (the upper bound on PrivBayes' degree k).
    /// </summary>
    /// <value>Maximum parents, defaulting to 3. Higher values allow more complex dependencies but increase computation.</value>
    /// <remarks>
    /// With differential privacy enabled, the degree actually used is the largest k up to this bound that
    /// keeps the network theta-useful (see <see cref="UsefulnessThreshold"/>), and every attribute after the
    /// first k gets exactly k parents, as in the paper's GreedyBayes.
    /// </remarks>
    public int MaxParents { get; set; } = 3;

    /// <summary>
    /// Gets or sets PrivBayes' usefulness threshold theta, used to choose the network degree k.
    /// </summary>
    /// <value>Defaults to 4, the value the paper recommends and uses in its experiments.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This decides how many parent columns each column may depend on. A higher threshold
    /// keeps the network simpler (fewer parents), so each probability table has fewer cells and gets less noise.
    /// </para>
    /// k is the largest value up to <see cref="MaxParents"/> with
    /// <c>n * eps2 / (2 * (d - k) * m^(k+1)) &gt;= theta</c>, where eps2 is the phase-2 budget and m the
    /// largest attribute domain. A larger theta favours fewer parents with less noise per cell. Zero or a
    /// negative value disables the rule and uses <see cref="MaxParents"/> as k.
    /// </remarks>
    public double UsefulnessThreshold { get; set; } = 4.0;

    /// <summary>
    /// Gets or sets the public lower bound of each column's domain, used for the equal-width bins.
    /// </summary>
    /// <value>Defaults to <c>null</c>, which takes the observed column minimum.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are the smallest values each numeric column can take, known without looking at
    /// the private data (for example 0 for an age). Together with <see cref="DomainUpperBounds"/> they set the bin
    /// edges, so the bins reveal nothing about the data itself.
    /// </para>
    /// PrivBayes assumes attribute domains are public. Without bounds the observed minimum and maximum of
    /// each numeric column define the bins, and those values are released without noise, outside the
    /// differential-privacy guarantee. Supply both bounds (one entry per column) for the full guarantee;
    /// values outside them fall into the end bins. Entries for categorical columns are ignored.
    /// Supplying only one of the two bounds, or an upper bound below its lower bound for a numeric column,
    /// throws <see cref="System.ArgumentException"/>.
    /// <para>
    /// A categorical column's domain is its <see cref="AiDotNet.NeuralNetworks.SyntheticData.ColumnMetadata.Categories"/>
    /// list, which is public: cells hold a category index (or a category name that is a number), and a
    /// declared category that never occurs in the data is still part of the domain. A categorical column
    /// that declares no categories falls back to the distinct values observed in the data. That fallback
    /// releases which values occur without noise and is outside the differential-privacy guarantee of
    /// PrivBayes; declare the categories for the full guarantee.
    /// </para>
    /// </remarks>
    public Vector<T>? DomainLowerBounds { get; set; }

    /// <summary>
    /// Gets or sets the public upper bound of each column's domain, used for the equal-width bins.
    /// </summary>
    /// <value>Defaults to <c>null</c>, which takes the observed column maximum.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are the largest values each numeric column can take, known without looking at
    /// the private data (for example 120 for an age). Supply them together with <see cref="DomainLowerBounds"/>,
    /// one entry per column; an upper bound below its lower bound is rejected.
    /// </para>
    /// See <see cref="DomainLowerBounds"/> for how the bounds are used.
    /// </remarks>
    public Vector<T>? DomainUpperBounds { get; set; }

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
