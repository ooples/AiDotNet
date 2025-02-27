namespace AiDotNet.Factories;

public static class FitnessCalculatorFactory
{
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