namespace AiDotNet.Helpers;

public static class OptimizerHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public static OptimizationResult<T> CreateOptimizationResult(
        ISymbolicModel<T> bestSolution,
        T bestFitness,
        List<T> fitnessHistory,
        List<Vector<T>> bestSelectedFeatures,
        OptimizationResult<T>.DatasetResult trainingResult,
        OptimizationResult<T>.DatasetResult validationResult,
        OptimizationResult<T>.DatasetResult testResult,
        FitDetectorResult<T> bestFitDetectionResult,
        int iterationCount)
    {
        return new OptimizationResult<T>
        {
            BestSolution = bestSolution,
            BestFitnessScore = bestFitness,
            Iterations = iterationCount,
            FitnessHistory = new Vector<T>([.. fitnessHistory]),
            SelectedFeatures = bestSelectedFeatures,
            TrainingResult = trainingResult,
            ValidationResult = validationResult,
            TestResult = testResult,
            FitDetectionResult = bestFitDetectionResult,
            CoefficientLowerBounds = Vector<T>.Empty(),
            CoefficientUpperBounds = Vector<T>.Empty()
        };
    }

    public static OptimizationResult<T>.DatasetResult CreateDatasetResult(
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

    public static List<int> GetSelectedFeatures(ISymbolicModel<T> solution)
    {
        if (solution == null)
        {
            throw new ArgumentNullException(nameof(solution));
        }

        var selectedFeatures = new List<int>();
        int featureCount = solution.FeatureCount;

        for (int i = 0; i < featureCount; i++)
        {
            if (solution.IsFeatureUsed(i))
            {
                selectedFeatures.Add(i);
            }
        }

        return selectedFeatures;
    }

    public static Matrix<T> SelectFeatures(Matrix<T> X, List<int> selectedFeatures)
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

    public static Matrix<T> SelectFeatures(Matrix<T> X, List<Vector<T>> selectedFeatures)
    {
        var selectedIndices = selectedFeatures.Select(v => GetFeatureIndex(v)).ToList();
        return X.SubMatrix(0, X.Rows - 1, selectedIndices);
    }

    private static int GetFeatureIndex(Vector<T> featureVector)
    {
        if (featureVector == null || featureVector.Length == 0)
        {
            throw new ArgumentException("Feature vector cannot be null or empty.", nameof(featureVector));
        }

        // Case 1: Binary feature vector (e.g., [0, 1, 0, 0])
        if (featureVector.All(v => NumOps.Equals(v, NumOps.Zero) || NumOps.Equals(v, NumOps.One)))
        {
            return featureVector.IndexOf(NumOps.One);
        }

        // Case 2: One-hot encoded vector (e.g., [0, 0, 1, 0])
        if (featureVector.Count(v => !NumOps.Equals(v, NumOps.Zero)) == 1)
        {
            return featureVector.IndexOfMax();
        }

        // Case 3: Continuous values (e.g., weights or importances)
        var threshold = NumOps.FromDouble(0.5); // Adjust this threshold as needed
        var maxValue = featureVector.Max();
        var normalizedVector = new Vector<T>(featureVector.Select(v => NumOps.Divide(v, maxValue)));

        for (int i = 0; i < normalizedVector.Length; i++)
        {
            if (NumOps.GreaterThan(normalizedVector[i], threshold))
            {
                return i;
            }
        }

        // If no value is above the threshold, return the index of the maximum value
        return featureVector.IndexOfMax();
    }

    public static OptimizationInputData<T> CreateOptimizationInputData(
        Matrix<T> xTrain, Vector<T> yTrain,
        Matrix<T> xVal, Vector<T> yVal,
        Matrix<T> xTest, Vector<T> yTest)
    {
        return new OptimizationInputData<T>
        {
            XTrain = xTrain,
            YTrain = yTrain,
            XVal = xVal,
            YVal = yVal,
            XTest = xTest,
            YTest = yTest
        };
    }

    public static T LineSearch(
        ISymbolicModel<T> currentSolution, 
        Vector<T> direction, 
        Vector<T> gradient, 
        OptimizationInputData<T> inputData,
        T initialStepSize,
        T? c1 = default,
        T? c2 = default)
    {
        c1 ??= NumOps.FromDouble(1e-4);
        c2 ??= NumOps.FromDouble(0.9);

        T alpha = initialStepSize;
        T alphaMax = NumOps.FromDouble(10.0); // Maximum step size

        T initialLoss = CalculateLoss(currentSolution, inputData);
        T initialGradientDotDirection = gradient.DotProduct(direction);

        while (NumOps.LessThanOrEquals(alpha, alphaMax))
        {
            var newCoefficients = currentSolution.Coefficients.Add(direction.Multiply(alpha));
            var newSolution = currentSolution.UpdateCoefficients(newCoefficients);
            T newLoss = CalculateLoss(newSolution, inputData);

            // Armijo condition (sufficient decrease)
            if (NumOps.LessThanOrEquals(newLoss, 
                NumOps.Add(initialLoss, 
                    NumOps.Multiply(NumOps.Multiply(c1, alpha), initialGradientDotDirection))))
            {
                var newGradient = CalculateGradient(newSolution, inputData.XTrain, inputData.YTrain);
                T newGradientDotDirection = newGradient.DotProduct(direction);

                // Curvature condition (Wolfe condition)
                if (NumOps.GreaterThanOrEquals(NumOps.Abs(newGradientDotDirection), 
                    NumOps.Multiply(c2, NumOps.Abs(initialGradientDotDirection))))
                {
                    return alpha;
                }
            }

            alpha = NumOps.Multiply(alpha, NumOps.FromDouble(0.5)); // Reduce step size
        }

        return initialStepSize; // Return initial step size if line search fails
    }

    private static T CalculateLoss(ISymbolicModel<T> solution, OptimizationInputData<T> inputData)
    {
        var predictions = solution.Predict(inputData.XTrain);
        return StatisticsHelper<T>.CalculateMeanSquaredError(inputData.YTrain, predictions);
    }

    private static Vector<T> CalculateGradient(ISymbolicModel<T> solution, Matrix<T> x, Vector<T> y)
    {
        var predictions = solution.Predict(x);
        var errors = predictions.Subtract(y);

        return x.Transpose().Multiply(errors).Divide(NumOps.FromDouble(x.Rows));
    }
}