namespace AiDotNet.Distributions;

/// <summary>
/// Defines a parametric probability distribution with learnable parameters.
/// </summary>
/// <remarks>
/// <para>
/// Parametric distributions are fully specified by a fixed set of parameters
/// (e.g., mean and variance for Normal, shape and rate for Gamma).
/// This interface provides methods for computing probability densities,
/// cumulative distributions, and gradients needed for gradient-based learning.
/// </para>
/// <para>
/// <b>For Beginners:</b> A parametric distribution is like a template for
/// describing uncertainty. For example, a Normal distribution is defined by
/// just two numbers: mean (center) and variance (spread). Once you know these
/// parameters, you know everything about the distribution.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ParametricDistribution")]
public interface IParametricDistribution<T>
{
    /// <summary>
    /// Gets the number of parameters that define this distribution.
    /// </summary>
    /// <remarks>
    /// For example: Normal has 2 (mean, variance), Gamma has 2 (shape, rate),
    /// Student-t has 3 (mean, scale, degrees of freedom).
    /// </remarks>
    int NumParameters { get; }

    /// <summary>
    /// Gets or sets the distribution parameters as a vector.
    /// </summary>
    /// <remarks>
    /// The interpretation of parameters varies by distribution:
    /// - Normal: [mean, variance]
    /// - Laplace: [location, scale]
    /// - Gamma: [shape, rate]
    /// </remarks>
    Vector<T> Parameters { get; set; }

    /// <summary>
    /// Gets the parameter names for this distribution.
    /// </summary>
    string[] ParameterNames { get; }

    /// <summary>
    /// Computes the probability density function (PDF) at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the density.</param>
    /// <returns>The probability density at x.</returns>
    T Pdf(T x);

    /// <summary>
    /// Computes the log probability density function at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the log density.</param>
    /// <returns>The log probability density at x.</returns>
    /// <remarks>
    /// Log PDF is numerically more stable than PDF for extreme values
    /// and is used in maximum likelihood estimation.
    /// </remarks>
    T LogPdf(T x);

    /// <summary>
    /// Computes the cumulative distribution function (CDF) at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the CDF.</param>
    /// <returns>The probability that a random variable is less than or equal to x.</returns>
    T Cdf(T x);

    /// <summary>
    /// Computes the inverse CDF (quantile function) for probability p.
    /// </summary>
    /// <param name="p">The probability (must be between 0 and 1).</param>
    /// <returns>The value x such that CDF(x) = p.</returns>
    T InverseCdf(T p);

    /// <summary>
    /// Gets the mean (expected value) of the distribution.
    /// </summary>
    T Mean { get; }

    /// <summary>
    /// Gets the variance of the distribution.
    /// </summary>
    T Variance { get; }

    /// <summary>
    /// Gets the standard deviation of the distribution.
    /// </summary>
    T StdDev { get; }

    /// <summary>
    /// Computes the gradient of the log PDF with respect to each parameter.
    /// </summary>
    /// <param name="x">The point at which to compute gradients.</param>
    /// <returns>Array of gradients, one per parameter.</returns>
    /// <remarks>
    /// These gradients are essential for gradient-based optimization
    /// in probabilistic models like NGBoost.
    /// </remarks>
    Vector<T> GradLogPdf(T x);

    /// <summary>
    /// Computes the Fisher Information Matrix for the distribution.
    /// </summary>
    /// <returns>The Fisher Information Matrix as a 2D array.</returns>
    /// <remarks>
    /// The Fisher Information Matrix measures the amount of information
    /// that an observable random variable carries about an unknown parameter.
    /// It's used in natural gradient descent.
    /// </remarks>
    Matrix<T> FisherInformation();

    /// <summary>
    /// Creates a copy of this distribution with the same parameters.
    /// </summary>
    IParametricDistribution<T> Clone();
}
