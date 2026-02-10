namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for AIM (Adaptive Iterative Mechanism), a marginal-based
/// differentially private synthetic data generation method.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// AIM generates synthetic data through iterative marginal measurements:
/// - <b>Marginal selection</b>: Uses the exponential mechanism to privately select informative marginals
/// - <b>Noisy measurement</b>: Measures selected marginals with calibrated Gaussian noise
/// - <b>Synthetic optimization</b>: Iteratively refines synthetic data to match measured marginals
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
///     MaxMarginalOrder = 2,    // Consider pairs of columns
///     NumIterations = 100
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
    /// Gets or sets the maximum marginal order to consider.
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
    /// Gets or sets the number of bins for discretizing continuous columns.
    /// </summary>
    /// <value>Number of bins, defaulting to 32.</value>
    public int NumBins { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of iterations for synthetic data optimization.
    /// </summary>
    /// <value>Number of iterations, defaulting to 100.</value>
    public int NumIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of marginals to select per iteration.
    /// </summary>
    /// <value>Marginals per iteration, defaulting to 5.</value>
    public int MarginalsPerIteration { get; set; } = 5;

    /// <summary>
    /// Gets or sets the learning rate for synthetic data optimization.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.1.</value>
    public double LearningRate { get; set; } = 0.1;
}
