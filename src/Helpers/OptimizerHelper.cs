namespace AiDotNet.Helpers;

public static class OptimizerHelper
{
    public static OptimizationResult<T> CreateOptimizationResult<T>(
        ISymbolicModel<T> bestSolution,
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
            BestSolution = bestSolution,
            BestFitnessScore = bestFitness,
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

    public static List<int> GetSelectedFeatures<T>(ISymbolicModel<T> solution)
    {
        if (solution == null)
        {
            throw new ArgumentNullException(nameof(solution));
        }

        var selectedFeatures = new List<int>();
        var numOps = MathHelper.GetNumericOperations<T>();

        // Assuming ISymbolicModel<T> has a property or method to get the number of features
        int featureCount = solution.FeatureCount;

        for (int i = 0; i < featureCount; i++)
        {
            // Assuming ISymbolicModel<T> has a method to check if a feature is used
            if (solution.IsFeatureUsed(i))
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

    public static Matrix<T> SelectFeatures<T>(Matrix<T> X, List<Vector<T>> selectedFeatures)
    {
        var selectedIndices = selectedFeatures.Select(v => GetFeatureIndex(v)).ToList();
        return X.SubMatrix(0, X.Rows - 1, selectedIndices);
    }

    private static int GetFeatureIndex<T>(Vector<T> featureVector)
    {
        if (featureVector == null || featureVector.Length == 0)
        {
            throw new ArgumentException("Feature vector cannot be null or empty.", nameof(featureVector));
        }

        var numOps = MathHelper.GetNumericOperations<T>();

        // Case 1: Binary feature vector (e.g., [0, 1, 0, 0])
        if (featureVector.All(v => numOps.Equals(v, numOps.Zero) || numOps.Equals(v, numOps.One)))
        {
            return featureVector.IndexOf(numOps.One);
        }

        // Case 2: One-hot encoded vector (e.g., [0, 0, 1, 0])
        if (featureVector.Count(v => !numOps.Equals(v, numOps.Zero)) == 1)
        {
            return featureVector.IndexOfMax();
        }

        // Case 3: Continuous values (e.g., weights or importances)
        var threshold = numOps.FromDouble(0.5); // Adjust this threshold as needed
        var maxValue = featureVector.Max();
        var normalizedVector = new Vector<T>(featureVector.Select(v => numOps.Divide(v, maxValue)));

        for (int i = 0; i < normalizedVector.Length; i++)
        {
            if (numOps.GreaterThan(normalizedVector[i], threshold))
            {
                return i;
            }
        }

        // If no value is above the threshold, return the index of the maximum value
        return featureVector.IndexOfMax();
    }

    public static void UpdateAndApplyBestSolution<T>(
        ModelResult<T> currentResult,
        ref ModelResult<T> bestResult,
        Matrix<T> XTrainSubset,
        Matrix<T> XTestSubset,
        Matrix<T> XValSubset,
        IFitnessCalculator<T> fitnessCalculator)
    {
        if (fitnessCalculator.IsBetterFitness(currentResult.Fitness, bestResult.Fitness))
        {
            bestResult.Solution = currentResult.Solution;
            bestResult.Fitness = currentResult.Fitness;
            bestResult.FitDetectionResult = currentResult.FitDetectionResult;
            bestResult.TrainingPredictions = currentResult.TrainingPredictions;
            bestResult.ValidationPredictions = currentResult.ValidationPredictions;
            bestResult.TestPredictions = currentResult.TestPredictions;
            bestResult.EvaluationData = currentResult.EvaluationData;
            bestResult.SelectedFeatures = currentResult.SelectedFeatures;
            bestResult.TrainingFeatures = XTrainSubset;
            bestResult.ValidationFeatures = XValSubset;
            bestResult.TestFeatures = XTestSubset;
        }
    }
}