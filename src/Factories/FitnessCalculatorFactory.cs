namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates fitness calculators for evaluating machine learning models.
/// </summary>
public static class FitnessCalculatorFactory
{
    /// <summary>
    /// Creates a fitness calculator of the specified type.
    /// </summary>
    /// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
    /// <typeparam name="TInput">The input data type for the fitness calculator.</typeparam>
    /// <typeparam name="TOutput">The output data type for the fitness calculator.</typeparam>
    /// <param name="type">The type of fitness calculator to create.</param>
    /// <returns>An implementation of IFitnessCalculator<T, TInput, TOutput> for the specified fitness calculator type.</returns>
    /// <exception cref="ArgumentException">Thrown when the requested fitness calculator type is not supported.</exception>
    public static IFitnessCalculator<T, TInput, TOutput> CreateFitnessCalculator<T, TInput, TOutput>(FitnessCalculatorType type)
    {
        return type switch
        {
            FitnessCalculatorType.MeanSquaredError => new MeanSquaredErrorFitnessCalculator<T, TInput, TOutput>(),
            FitnessCalculatorType.MeanAbsoluteError => new MeanAbsoluteErrorFitnessCalculator<T, TInput, TOutput>(),
            FitnessCalculatorType.RSquared => new RSquaredFitnessCalculator<T, TInput, TOutput>(),
            FitnessCalculatorType.AdjustedRSquared => new AdjustedRSquaredFitnessCalculator<T, TInput, TOutput>(),
            FitnessCalculatorType.OrdinalRegressionLoss => new OrdinalRegressionLossFitnessCalculator<T, TInput, TOutput>(),
            FitnessCalculatorType.HuberLoss => new HuberLossFitnessCalculator<T, TInput, TOutput>(),
            FitnessCalculatorType.RootMeanSquaredError => new RootMeanSquaredErrorFitnessCalculator<T, TInput, TOutput>(),
            FitnessCalculatorType.ExponentialLoss => new ExponentialLossFitnessCalculator<T, TInput, TOutput>(),
            FitnessCalculatorType.Custom => throw new ArgumentException("Custom fitness calculators must be created directly, not through the factory."),
            _ => throw new ArgumentException($"Unsupported fitness calculator type: {type}"),
        };
    }
}
