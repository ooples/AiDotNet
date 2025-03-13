namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates fitness calculators for evaluating machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A fitness calculator measures how well your AI model is performing by comparing 
/// its predictions to the actual correct answers. Different calculators use different mathematical methods 
/// to measure this performance.
/// </para>
/// <para>
/// This factory class helps you create the right type of calculator without needing to know the implementation details.
/// Think of it like ordering a specific tool from a catalog - you just specify what you need, and the factory provides it.
/// </para>
/// </remarks>
public static class FitnessCalculatorFactory
{
    /// <summary>
    /// Creates a fitness calculator of the specified type.
    /// </summary>
    /// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
    /// <param name="type">The type of fitness calculator to create.</param>
    /// <returns>An implementation of IFitnessCalculator<T> for the specified fitness calculator type.</returns>
    /// <exception cref="ArgumentException">Thrown when the requested fitness calculator type is not supported.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The generic type parameter T (typically float or double) determines what kind of 
    /// number format will be used in calculations. Float uses less memory but has less precision than double.
    /// </para>
    /// <para>
    /// Available calculator types include:
    /// <list type="bullet">
    /// <item><description>MeanSquaredError: Emphasizes larger errors by squaring them</description></item>
    /// <item><description>MeanAbsoluteError: Treats all errors equally regardless of size</description></item>
    /// <item><description>RSquared: Measures how well your model explains the variation in data (0-1 scale)</description></item>
    /// <item><description>AdjustedRSquared: Like RSquared but accounts for the number of features in your model</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IFitnessCalculator<T> CreateFitnessCalculator<T>(FitnessCalculatorType type)
    {
        return type switch
        {
            FitnessCalculatorType.MeanSquaredError => new MeanSquaredErrorFitnessCalculator<T>(),
            FitnessCalculatorType.MeanAbsoluteError => new MeanAbsoluteErrorFitnessCalculator<T>(),
            FitnessCalculatorType.RSquared => new RSquaredFitnessCalculator<T>(),
            FitnessCalculatorType.AdjustedRSquared => new AdjustedRSquaredFitnessCalculator<T>(),
            _ => throw new ArgumentException($"Unsupported fitness calculator type: {type}"),
        };
    }
}