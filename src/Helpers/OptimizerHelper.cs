namespace AiDotNet.Helpers;

/// <summary>
/// Helper class that provides optimization-related functionality for machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public static class OptimizerHelper<T>
{
    /// <summary>
    /// Provides operations for the numeric type T (like addition, multiplication, etc.).
    /// </summary>
    /// <remarks>
    /// For Beginners: This is a utility that helps perform math operations regardless of 
    /// whether you're using integers, floats, doubles, or other numeric types.
    /// </remarks>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a result object containing all information about an optimization process.
    /// </summary>
    /// <param name="bestSolution">The best model found during optimization.</param>
    /// <param name="bestFitness">The fitness score of the best solution (lower is better).</param>
    /// <param name="fitnessHistory">List of fitness scores throughout the optimization process.</param>
    /// <param name="bestSelectedFeatures">List of feature vectors that were selected as most important.</param>
    /// <param name="trainingResult">Results from evaluating the model on training data.</param>
    /// <param name="validationResult">Results from evaluating the model on validation data.</param>
    /// <param name="testResult">Results from evaluating the model on test data.</param>
    /// <param name="bestFitDetectionResult">Information about model fit quality (underfitting/overfitting).</param>
    /// <param name="iterationCount">Number of iterations the optimization process ran for.</param>
    /// <returns>A complete optimization result object.</returns>
    /// <remarks>
    /// For Beginners: This method packages all the results from training an AI model into one 
    /// organized container. Think of it like a report card that shows how well your model performed,
    /// what features it found important, and whether it's a good fit for your data.
    /// </remarks>
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

    /// <summary>
    /// Creates a result object containing evaluation metrics for a specific dataset (training, validation, or test).
    /// </summary>
    /// <param name="predictions">The model's predictions for this dataset.</param>
    /// <param name="errorStats">Statistics about prediction errors (like MSE, MAE).</param>
    /// <param name="actualBasicStats">Basic statistics about the actual values (min, max, mean, etc.).</param>
    /// <param name="predictedBasicStats">Basic statistics about the predicted values.</param>
    /// <param name="predictionStats">Advanced statistics about prediction quality.</param>
    /// <param name="features">The feature matrix (X) used for this dataset.</param>
    /// <param name="y">The target values for this dataset.</param>
    /// <returns>A dataset result object containing all evaluation metrics.</returns>
    /// <remarks>
    /// For Beginners: This method creates a detailed report about how well your model performed on a 
    /// specific set of data. It includes various measurements of accuracy and error, as well as 
    /// information about the data itself. This helps you understand where your model is doing well 
    /// and where it might need improvement.
    /// </remarks>
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

    /// <summary>
    /// Identifies which features are being used by a model.
    /// </summary>
    /// <param name="solution">The model to analyze.</param>
    /// <returns>A list of indices representing the features used by the model.</returns>
    /// <remarks>
    /// For Beginners: Features are the individual pieces of information your model uses to make predictions
    /// (like "age", "height", etc.). This method checks which features your model is actually using,
    /// since some models might ignore certain features if they're not helpful for predictions.
    /// </remarks>
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

    /// <summary>
    /// Creates a new matrix containing only the selected features from the original data.
    /// </summary>
    /// <param name="X">The original feature matrix.</param>
    /// <param name="selectedFeatures">List of indices for features to keep.</param>
    /// <returns>A new matrix with only the selected features.</returns>
    /// <remarks>
    /// For Beginners: If your model only uses certain features, this method creates a simplified
    /// version of your data that includes just those important features. This can make your model
    /// faster and sometimes more accurate by focusing only on what matters.
    /// </remarks>
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

    /// <summary>
    /// Creates a new matrix containing only the selected features from the original data.
    /// </summary>
    /// <param name="X">The original feature matrix.</param>
    /// <param name="selectedFeatures">List of feature vectors representing important features.</param>
    /// <returns>A new matrix with only the selected features.</returns>
    /// <remarks>
    /// For Beginners: This is similar to the other SelectFeatures method, but it works with a different
    /// way of representing which features are important. Instead of simple indices, it uses vectors
    /// that might contain more complex information about feature importance.
    /// </remarks>
    public static Matrix<T> SelectFeatures(Matrix<T> X, List<Vector<T>> selectedFeatures)
    {
        var selectedIndices = selectedFeatures.Select(v => GetFeatureIndex(v)).ToList();
        return X.SubMatrix(0, X.Rows - 1, selectedIndices);
    }

    /// <summary>
    /// Determines which feature is represented by a feature vector.
    /// </summary>
    /// <param name="featureVector">A vector representing a feature.</param>
    /// <returns>The index of the feature represented by this vector.</returns>
    /// <remarks>
    /// For Beginners: This helper method interprets different ways that important features might be
    /// represented in your model. It handles several common formats and converts them to a simple
    /// feature index (like "feature #3").
    /// </remarks>
    private static int GetFeatureIndex(Vector<T> featureVector)
    {
        if (featureVector == null || featureVector.Length == 0)
        {
            throw new ArgumentException("Feature vector cannot be null or empty.", nameof(featureVector));
        }

        // Case 1: Binary feature vector (e.g., [0, 1, 0, 0])
        if (featureVector.All(v => _numOps.Equals(v, _numOps.Zero) || _numOps.Equals(v, _numOps.One)))
        {
            return featureVector.IndexOf(_numOps.One);
        }

        // Case 2: One-hot encoded vector (e.g., [0, 0, 1, 0])
        if (featureVector.Count(v => !_numOps.Equals(v, _numOps.Zero)) == 1)
        {
            return featureVector.IndexOfMax();
        }

        // Case 3: Continuous values (e.g., weights or importances)
        var threshold = _numOps.FromDouble(0.5); // Adjust this threshold as needed
        var maxValue = featureVector.Max();
        var normalizedVector = new Vector<T>(featureVector.Select(v => _numOps.Divide(v, maxValue)));

        for (int i = 0; i < normalizedVector.Length; i++)
        {
            if (_numOps.GreaterThan(normalizedVector[i], threshold))
            {
                return i;
            }
        }

        // If no value is above the threshold, return the index of the maximum value
        return featureVector.IndexOfMax();
    }

    /// <summary>
    /// Creates a data container for optimization algorithms with training, validation, and test datasets.
    /// </summary>
    /// <param name="xTrain">Feature matrix for training the model.</param>
    /// <param name="yTrain">Target values for training the model.</param>
    /// <param name="xVal">Feature matrix for validating the model during training.</param>
    /// <param name="yVal">Target values for validation.</param>
    /// <param name="xTest">Feature matrix for final testing of the model.</param>
    /// <param name="yTest">Target values for testing.</param>
    /// <returns>A structured container with all datasets needed for optimization.</returns>
    /// <remarks>
    /// For Beginners: This method organizes your data into three separate sets:
    /// 1. Training data - Used to teach your model patterns (like studying for a test)
    /// 2. Validation data - Used to check progress during training (like practice questions)
    /// 3. Test data - Used for final evaluation (like the actual test)
    /// 
    /// This separation helps ensure your model can generalize to new data it hasn't seen before.
    /// </remarks>
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

    /// <summary>
    /// Performs a line search to find the optimal step size for gradient-based optimization.
    /// </summary>
    /// <param name="currentSolution">The current model being optimized.</param>
    /// <param name="direction">The direction vector to move in parameter space.</param>
    /// <param name="gradient">The gradient vector indicating the direction of steepest ascent.</param>
    /// <param name="inputData">The training, validation, and test datasets.</param>
    /// <param name="initialStepSize">Starting step size for the search.</param>
    /// <param name="c1">Armijo condition parameter (controls sufficient decrease).</param>
    /// <param name="c2">Wolfe condition parameter (controls curvature).</param>
    /// <returns>The optimal step size to use for updating model parameters.</returns>
    /// <remarks>
    /// For Beginners: Line search is like finding the right "step size" when walking downhill.
    /// If you take steps that are too small, you'll waste time. If you take steps that are too big,
    /// you might overshoot and go uphill again. This method finds the "just right" step size
    /// to help your model improve as quickly as possible without overshooting.
    /// 
    /// The Armijo and Wolfe conditions are mathematical checks that ensure we're making good progress
    /// in the right direction.
    /// </remarks>
    public static T LineSearch(
        ISymbolicModel<T> currentSolution, 
        Vector<T> direction, 
        Vector<T> gradient, 
        OptimizationInputData<T> inputData,
        T initialStepSize,
        T? c1 = default,
        T? c2 = default)
    {
        c1 ??= _numOps.FromDouble(1e-4);  // Default Armijo condition parameter
        c2 ??= _numOps.FromDouble(0.9);   // Default Wolfe condition parameter

        T alpha = initialStepSize;
        T alphaMax = _numOps.FromDouble(10.0); // Maximum step size

        T initialLoss = CalculateLoss(currentSolution, inputData);
        T initialGradientDotDirection = gradient.DotProduct(direction);

        while (_numOps.LessThanOrEquals(alpha, alphaMax))
        {
            var newCoefficients = currentSolution.Coefficients.Add(direction.Multiply(alpha));
            var newSolution = currentSolution.UpdateCoefficients(newCoefficients);
            T newLoss = CalculateLoss(newSolution, inputData);

            // Armijo condition (sufficient decrease)
            if (_numOps.LessThanOrEquals(newLoss, 
                _numOps.Add(initialLoss, 
                    _numOps.Multiply(_numOps.Multiply(c1, alpha), initialGradientDotDirection))))
            {
                var newGradient = CalculateGradient(newSolution, inputData.XTrain, inputData.YTrain);
                T newGradientDotDirection = newGradient.DotProduct(direction);

                // Curvature condition (Wolfe condition)
                if (_numOps.GreaterThanOrEquals(_numOps.Abs(newGradientDotDirection), 
                    _numOps.Multiply(c2, _numOps.Abs(initialGradientDotDirection))))
                {
                    return alpha;
                }
            }

            alpha = _numOps.Multiply(alpha, _numOps.FromDouble(0.5)); // Reduce step size
        }

        return initialStepSize; // Return initial step size if line search fails
    }

    /// <summary>
    /// Calculates the loss (error) for a model on the training data.
    /// </summary>
    /// <param name="solution">The model to evaluate.</param>
    /// <param name="inputData">The dataset containing training data.</param>
    /// <returns>The mean squared error of the model's predictions.</returns>
    /// <remarks>
    /// For Beginners: Loss is a measure of how wrong your model's predictions are.
    /// Lower loss means better predictions. This method calculates the "Mean Squared Error",
    /// which is the average of the squared differences between predictions and actual values.
    /// Squaring the differences ensures that both underestimates and overestimates count as errors.
    /// </remarks>
    private static T CalculateLoss(ISymbolicModel<T> solution, OptimizationInputData<T> inputData)
    {
        var predictions = solution.Predict(inputData.XTrain);
        return StatisticsHelper<T>.CalculateMeanSquaredError(inputData.YTrain, predictions);
    }

    /// <summary>
    /// Calculates the gradient of the loss function with respect to model parameters.
    /// </summary>
    /// <param name="solution">The model whose gradient is being calculated.</param>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The target values.</param>
    /// <returns>A vector representing the gradient (direction of steepest increase in loss).</returns>
    /// <remarks>
    /// For Beginners: The gradient is like a compass that points in the direction where the error
    /// increases most rapidly. By going in the opposite direction, we can reduce the error.
    /// 
    /// This method calculates this "compass direction" for our model parameters. For linear models,
    /// this is calculated as X^T * (predictions - actual) / n, where X^T is the transpose of the
    /// feature matrix, and n is the number of data points.
    /// </remarks>
    private static Vector<T> CalculateGradient(ISymbolicModel<T> solution, Matrix<T> x, Vector<T> y)
    {
        var predictions = solution.Predict(x);
        var errors = predictions.Subtract(y);

        return x.Transpose().Multiply(errors).Divide(_numOps.FromDouble(x.Rows));
    }
}