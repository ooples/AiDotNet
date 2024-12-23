namespace AiDotNet.Factories;

public static class FitnessCalculatorFactory
{
    public static IFitnessCalculator<T> CreateFitnessCalculator<T>(FitnessCalculatorType type)
    {
        switch (type)
        {
            case FitnessCalculatorType.MeanSquaredError:
                return new MSEFitnessCalculator<T>();
            case FitnessCalculatorType.MeanAbsoluteError:
                return new MAEFitnessCalculator<T>();
            case FitnessCalculatorType.RSquared:
                return new RSquaredFitnessCalculator<T>();
            case FitnessCalculatorType.AdjustedRSquared:
                return new AdjustedRSquaredFitnessCalculator<T>();
            default:
                throw new ArgumentException($"Unsupported fitness calculator type: {type}");
        }
    }
}