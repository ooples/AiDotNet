namespace AiDotNet.Helpers;

public static class OptimizerHelper
{
    public static OptimizationResult<T> CreateOptimizationResult<T>(
        Vector<T> bestSolution,
        T bestIntercept,
        T bestFitness,
        List<T> fitnessHistory,
        List<Vector<T>> bestSelectedFeatures,
        OptimizationResult<T>.DatasetResult trainingResult,
        OptimizationResult<T>.DatasetResult validationResult,
        OptimizationResult<T>.DatasetResult testResult,
        FitDetectorResult<T> bestFitDetectionResult,
        int iterationCount,
        INumericOperations<T> numOps)
    {
        return new OptimizationResult<T>
        {
            BestCoefficients = bestSolution,
            BestIntercept = bestIntercept,
            FitnessScore = bestFitness,
            Iterations = iterationCount,
            FitnessHistory = new Vector<T>([.. fitnessHistory], numOps),
            SelectedFeatures = bestSelectedFeatures,
            TrainingResult = trainingResult,
            ValidationResult = validationResult,
            TestResult = testResult,
            FitDetectionResult = bestFitDetectionResult,
            CoefficientLowerBounds = Vector<T>.Empty(),
            CoefficientUpperBounds = Vector<T>.Empty()
        };
    }

    public static OptimizationResult<T>.DatasetResult CreateDatasetResult<T>(
        Vector<T>? predictions,
        ErrorStats<T>? errorStats,
        BasicStats<T>? actualBasicStats,
        BasicStats<T>? predictedBasicStats,
        PredictionStats<T>? predictionStats,
        Matrix<T>? features,
        Vector<T> y)
    {
        return new OptimizationResult<T>.DatasetResult
        {
            Predictions = predictions ?? Vector<T>.Empty(),
            ErrorStats = errorStats ?? ErrorStats<T>.Empty(),
            ActualBasicStats = actualBasicStats ?? BasicStats<T>.Empty(),
            PredictedBasicStats = predictedBasicStats ?? BasicStats<T>.Empty(),
            PredictionStats = predictionStats ?? PredictionStats<T>.Empty(),
            X = features ?? Matrix<T>.Empty(),
            Y = y
        };
    }

    public static List<int> GetSelectedFeatures<T>(Vector<T> solution)
    {
        if (solution == null)
        {
            throw new ArgumentNullException(nameof(solution));
        }

        var selectedFeatures = new List<int>();
        for (int i = 0; i < solution.Length; i++)
        {
            if (!(solution[i] ?? default)?.Equals(default(T)) ?? true)
            {
                selectedFeatures.Add(i);
            }
        }

        return selectedFeatures;
    }

    public static Matrix<T> SelectFeatures<T>(Matrix<T> X, List<int> selectedFeatures)
    {
        var selectedX = new Matrix<T>(X.Rows, selectedFeatures.Count);
        for (int i = 0; i < selectedFeatures.Count; i++)
        {
            var featureIndex = selectedFeatures[i];
            for (int j = 0; j < X.Rows; j++)
            {
                selectedX[j, i] = X[j, featureIndex];
            }
        }

        return selectedX;
    }
}