namespace AiDotNet.FitnessCalculators;

public class OrdinalRegressionLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    private readonly int? _numClasses;

    public OrdinalRegressionLossFitnessCalculator(int? numberOfClassifications = null, DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
        _numClasses = numberOfClassifications;
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        if (_numClasses.HasValue)
        {
            return NeuralNetworkHelper<T>.OrdinalRegressionLoss(dataSet.Predicted, dataSet.Actual, _numClasses.Value);
        }
        else
        {
            // Default process when numClasses is not provided
            // Potential way to use this loss function with other problems like regression
            return DefaultLossCalculation(dataSet);
        }
    }

    private T DefaultLossCalculation(DataSetStats<T> dataSet)
    {
        // Determine the type of problem based on the data
        if (IsClassificationProblem(dataSet))
        {
            // For classification, use the number of unique values in Actual as numClasses
            int numClasses = dataSet.Actual.Distinct().Count();
            return NeuralNetworkHelper<T>.OrdinalRegressionLoss(dataSet.Predicted, dataSet.Actual, numClasses);
        }
        else
        {
            // For regression or other problems, use a different loss calculation
            return NeuralNetworkHelper<T>.MeanSquaredError(dataSet.Predicted, dataSet.Actual);
        }
    }

    private bool IsClassificationProblem(DataSetStats<T> dataSet)
    {
        // Get unique values
        var uniqueValues = dataSet.Actual.Distinct().ToList();
        int uniqueCount = uniqueValues.Count;
        int totalCount = dataSet.Actual.Length;

        // Check if all values are integers (or can be parsed as integers)
        bool allIntegers = uniqueValues.All(v => int.TryParse(v?.ToString(), out _));

        // Check if the number of unique values is small relative to the total number of samples
        bool fewUniqueValues = uniqueCount <= Math.Min(10, Math.Sqrt(totalCount));

        // Check if values are evenly spaced (for ordinal data)
        bool evenlySpaced = false;
        if (allIntegers && uniqueCount > 1)
        {
            var sortedValues = uniqueValues.Select(v => Convert.ToInt32(v)).OrderBy(v => v).ToList();
            int commonDifference = sortedValues[1] - sortedValues[0];
            evenlySpaced = sortedValues.Zip(sortedValues.Skip(1), (a, b) => b - a)
                                       .All(diff => diff == commonDifference);
        }

        // Check if values are within a specific range (e.g., 0 to 1 for probabilities)
        bool withinProbabilityRange = uniqueValues.All(v => _numOps.GreaterThanOrEquals(v, _numOps.Zero) && _numOps.LessThanOrEquals(v, _numOps.One));

        // Combine all checks
        return (allIntegers && fewUniqueValues) || evenlySpaced || withinProbabilityRange;
    }
}