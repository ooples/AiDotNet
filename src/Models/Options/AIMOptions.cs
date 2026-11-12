namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for AIM (Adaptive Iterative Mechanism), a marginal-based
/// differentially private synthetic data generation method.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// AIM (McKenna et al. 2022) generates synthetic data through adaptive, iterative marginal measurements
/// under zero-concentrated differential privacy:
/// - <b>Marginal selection</b>: the exponential mechanism picks the worst-approximated workload marginal
/// - <b>Noisy measurement</b>: the Gaussian mechanism measures it; the noise is annealed as the model improves
/// - <b>Model estimation</b>: Private-PGM fits a graphical model to every noisy measurement
/// </para>
/// <para>
/// <b>For Beginners:</b> AIM is a non-neural approach to private data synthesis:
///
/// 1. Pick important "statistics" about the data (marginals = histograms of 1-3 columns)
/// 2. Measure them with added noise (for privacy)
/// 3. Create fake data that matches those noisy statistics
///
/// Unlike GAN/VAE models, AIM uses mathematical optimization (not deep learning),
/// which often works better for smaller datasets and provides formal privacy guarantees.
///
/// Example:
/// <code>
/// var options = new AIMOptions&lt;double&gt;
/// {
///     Epsilon = 1.0,           // Privacy budget
///     MaxMarginalOrder = 2,    // Workload: all pairs of columns
///     NumIterations = 1000     // Private-PGM iterations per estimate
/// };
/// var aim = new AIMGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "AIM: An Adaptive and Iterative Mechanism for Differentially Private
/// Synthetic Data" (McKenna et al., 2022)
/// </para>
/// </remarks>
public class AIMOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the total privacy budget (epsilon).
    /// </summary>
    /// <value>Epsilon, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls privacy:
    /// - Lower epsilon = more privacy, less accuracy
    /// - Higher epsilon = less privacy, more accuracy
    /// - 1.0 is a common starting point for moderate privacy
    /// </para>
    /// </remarks>
    public double Epsilon { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the order of the workload marginals (every combination of this many columns).
    /// </summary>
    /// <value>Maximum order, defaulting to 2 (pairwise marginals).</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The order determines how many columns are considered together:
    /// - Order 1: Histograms of individual columns
    /// - Order 2: Joint histograms of pairs of columns
    /// - Order 3: Joint histograms of triples (expensive)
    /// Higher orders capture more relationships but require more privacy budget.
    /// </para>
    /// </remarks>
    public int MaxMarginalOrder { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of equal-width bins for discretizing numeric columns.
    /// </summary>
    /// <value>Number of bins, defaulting to 32.</value>
    public int NumBins { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of Private-PGM mirror-descent iterations per model estimate.
    /// </summary>
    /// <value>Defaults to 1000, private-pgm's default for AIM.</value>
    /// <remarks>
    /// Each round re-estimates the graphical model from all measurements so far, warm-started from the
    /// previous estimate. Fewer iterations make fitting faster at the cost of a less converged model.
    /// </remarks>
    public int NumIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets delta of the (epsilon, delta) guarantee.
    /// </summary>
    /// <value>Defaults to 1e-9, the value used in the paper's experiments.</value>
    public double Delta { get; set; } = 1e-9;

    /// <summary>
    /// Gets or sets the number of rounds T used to set the initial noise scales.
    /// </summary>
    /// <value>Defaults to <c>null</c>, which uses the paper's T = 16 d for d columns.</value>
    /// <remarks>
    /// T only sets the starting scales <c>sigma_0 = sqrt(T / (2 alpha rho))</c> and
    /// <c>epsilon_0 = sqrt(8 (1 - alpha) rho / T)</c>; AIM anneals them and stops when the budget is spent.
    /// </remarks>
    public int? Rounds { get; set; }

    /// <summary>
    /// Gets or sets alpha, the fraction of each round's zCDP budget spent on measurement (the rest pays
    /// for selection).
    /// </summary>
    /// <value>Defaults to 0.9, the paper's value.</value>
    public double MeasurementBudgetFraction { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the maximum size of the graphical model's junction tree, in megabytes.
    /// </summary>
    /// <value>Defaults to 80, the paper's value.</value>
    /// <remarks>
    /// After spending a fraction f of the budget, only marginals that keep the junction tree within
    /// <c>f * MaxModelSizeMb</c> can be selected.
    /// </remarks>
    public double MaxModelSizeMb { get; set; } = 80.0;

    /// <summary>
    /// Gets or sets the largest number of cells a workload marginal may have.
    /// </summary>
    /// <value>Defaults to 10000, private-pgm's default for AIM.</value>
    public int MaxCells { get; set; } = 10000;

    /// <summary>
    /// Gets or sets the public lower bound of each column's domain, used for the equal-width bins.
    /// </summary>
    /// <value>Defaults to <c>null</c>, which takes the observed column minimum.</value>
    /// <remarks>
    /// AIM assumes attribute domains are public. Without bounds the observed minimum and maximum of each
    /// numeric column define the bins, and those values are released without noise, outside the
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
    /// AIM; declare the categories for the full guarantee.
    /// </para>
    /// </remarks>
    public Vector<T>? DomainLowerBounds { get; set; }

    /// <summary>
    /// Gets or sets the public upper bound of each column's domain, used for the equal-width bins.
    /// </summary>
    /// <value>Defaults to <c>null</c>, which takes the observed column maximum.</value>
    /// <remarks>See <see cref="DomainLowerBounds"/>.</remarks>
    public Vector<T>? DomainUpperBounds { get; set; }
}
