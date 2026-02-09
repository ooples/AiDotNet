namespace AiDotNet.Distributions;

/// <summary>
/// Extends parametric distributions with sampling capabilities.
/// </summary>
/// <remarks>
/// <para>
/// Sampling distributions support generating random variates from the distribution.
/// This is essential for Monte Carlo methods, simulation, and Bayesian inference.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sampling means generating random numbers that follow
/// a specific pattern (distribution). For example, if you sample from a Normal
/// distribution with mean 0 and variance 1, most samples will be close to 0,
/// and samples far from 0 (like 3 or -3) will be rare.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ISamplingDistribution<T> : IParametricDistribution<T>
{
    /// <summary>
    /// Generates a single random sample from the distribution.
    /// </summary>
    /// <param name="random">The random number generator to use.</param>
    /// <returns>A random sample from the distribution.</returns>
    T Sample(Random random);

    /// <summary>
    /// Generates multiple random samples from the distribution.
    /// </summary>
    /// <param name="random">The random number generator to use.</param>
    /// <param name="count">The number of samples to generate.</param>
    /// <returns>An array of random samples.</returns>
    Vector<T> Sample(Random random, int count);

    /// <summary>
    /// Generates a single random sample using a default random number generator.
    /// </summary>
    /// <returns>A random sample from the distribution.</returns>
    T Sample();

    /// <summary>
    /// Generates multiple random samples using a default random number generator.
    /// </summary>
    /// <param name="count">The number of samples to generate.</param>
    /// <returns>An array of random samples.</returns>
    Vector<T> Sample(int count);
}
