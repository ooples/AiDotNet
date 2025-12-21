namespace AiDotNet.Extensions;

/// <summary>
/// Provides extension methods for the Random class to generate numbers with specific distributions.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class adds new capabilities to .NET's built-in Random class.
/// While the standard Random class can generate uniform random numbers (where every value has an equal chance),
/// AI and machine learning often need different types of random numbers that follow specific patterns or distributions.
/// </para>
/// </remarks>
public static class RandomExtensions
{
    /// <summary>
    /// Generates a random number from a Gaussian (normal) distribution with mean 0 and standard deviation 1.
    /// </summary>
    /// <param name="random">The Random object to extend.</param>
    /// <returns>A random double value from a standard normal distribution.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> While regular random numbers (from Random.NextDouble()) give you any value between 0 and 1
    /// with equal probability, Gaussian random numbers follow the famous "bell curve" pattern.
    /// </para>
    /// <para>
    /// In a Gaussian distribution:
    /// <list type="bullet">
    ///   <item><description>Values near the mean (0 in this case) are more likely to occur</description></item>
    ///   <item><description>Values far from the mean are less likely to occur</description></item>
    ///   <item><description>The standard deviation (1 in this case) controls how spread out the values are</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// This method uses the Box-Muller transform, a mathematical technique to convert uniform random numbers
    /// into Gaussian random numbers.
    /// </para>
    /// <para>
    /// Gaussian distributions are extremely important in AI and machine learning because:
    /// <list type="bullet">
    ///   <item><description>Many natural phenomena follow this distribution</description></item>
    ///   <item><description>They're used to initialize weights in neural networks</description></item>
    ///   <item><description>They're used in many algorithms like Gaussian processes and variational autoencoders</description></item>
    ///   <item><description>They're essential for adding noise in techniques like simulated annealing</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// If you need a Gaussian distribution with a different mean and standard deviation,
    /// you can transform the result: <c>mean + standardDeviation * NextGaussian()</c>
    /// </para>
    /// </remarks>
    public static double NextGaussian(this Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();

        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}
