namespace AiDotNet.Helpers;

/// <summary>
/// Helper class that provides optimization-related functionality for machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector<T>, Tensor<T>).</typeparam>
public static class OptimizerHelper<T, TInput, TOutput>
{
    /// <summary>
    /// Provides operations for the numeric type T (like addition, multiplication, etc.).
    /// </summary>
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
    public static OptimizationResult<T, TInput, TOutput> CreateOptimizationResult(
        IFullModel<T, TInput, TOutput> bestSolution,
        T bestFitness,
        List<T> fitnessHistory,
        List<Vector<T>> bestSelectedFeatures,
        OptimizationResult<T, TInput, TOutput>.DatasetResult trainingResult,
        OptimizationResult<T, TInput, TOutput>.DatasetResult validationResult,
        OptimizationResult<T, TInput, TOutput>.DatasetResult testResult,
        FitDetectorResult<T> bestFitDetectionResult,
        int iterationCount)
    {
        return new OptimizationResult<T, TInput, TOutput>
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
    public static OptimizationResult<T, TInput, TOutput>.DatasetResult CreateDatasetResult(
        TOutput predictions,
        ErrorStats<T>? errorStats,
        BasicStats<T>? actualBasicStats,
        BasicStats<T>? predictedBasicStats,
        PredictionStats<T>? predictionStats,
        TInput features,
        TOutput y)
    {
        var modelType = predictionStats?.ModelType ?? ModelType.None;
        return new OptimizationResult<T, TInput, TOutput>.DatasetResult(modelType)
        {
            Predictions = predictions,
            ErrorStats = errorStats ?? ErrorStats<T>.Empty(),
            ActualBasicStats = actualBasicStats ?? BasicStats<T>.Empty(),
            PredictedBasicStats = predictedBasicStats ?? BasicStats<T>.Empty(),
            PredictionStats = predictionStats ?? PredictionStats<T>.Empty(),
            X = features,
            Y = y
        };
    }

    /// <summary>
    /// Creates a new input containing only the selected features from the original data.
    /// </summary>
    /// <param name="X">The original feature input (Matrix<T> or Tensor<T>).</param>
    /// <param name="selectedFeatures">List of indices for features to keep.</param>
    /// <returns>A new input with only the selected features.</returns>
    public static TInput SelectFeatures(TInput X, List<int> selectedFeatures)
    {
        if (X is Matrix<T> matrix)
        {
            return (TInput)(object)SelectFeaturesMatrix(matrix, selectedFeatures);
        }
        else if (X is Tensor<T> tensor)
        {
            return (TInput)(object)SelectFeaturesTensor(tensor, selectedFeatures);
        }
        else
        {
            throw new ArgumentException("Unsupported input type for feature selection", nameof(X));
        }
    }

    /// <summary>
    /// Creates a new input with only the specified features.
    /// </summary>
    /// <param name="input">The original input data.</param>
    /// <param name="selectedFeatures">The features to include in the new input.</param>
    /// <returns>A new input containing only the selected features.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a reduced version of the input data that contains only the selected features.
    /// It supports different input types like Matrix<T> and Tensor<T>.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a simplified version of your dataset.
    /// 
    /// Imagine you have a large dataset with many features (columns), but you only want to use some of them:
    /// - This method builds a new dataset using only the features you've selected
    /// - The resulting dataset has fewer columns but the same number of rows
    /// - This is useful for feature selection, which can improve model performance by focusing on the most relevant data
    /// 
    /// For example, if your original dataset had 100 features but you only want to use 10 of them,
    /// this method would create a new dataset with just those 10 features.
    /// </para>
    /// </remarks>
    public static TInput SelectFeatures(TInput input, IEnumerable<int> selectedFeatureIndices)
    {
        var indices = selectedFeatureIndices.ToList();

        if (input is Matrix<T> matrix)
        {
            // Create a new matrix with selected columns from THIS matrix
            var selectedColumns = indices.Select(i => matrix.GetColumn(i)).ToArray();
            return (TInput)(object)Matrix<T>.FromColumns(selectedColumns);
        }
        else if (input is Tensor<T> tensor)
        {
            // Check indices are in bounds
            foreach (var idx in indices)
            {
                if (idx < 0 || idx >= tensor.Shape[1])
                    throw new ArgumentOutOfRangeException(nameof(selectedFeatureIndices),
                        $"Feature index {idx} is out of range for tensor with {tensor.Shape[1]} columns");
            }

            int rows = tensor.Shape[0];
            int cols = indices.Count;

            // Create a new tensor with selected columns from THIS tensor
            var result = new Tensor<T>([rows, cols]);

            for (int newCol = 0; newCol < cols; newCol++)
            {
                int origCol = indices[newCol];
                for (int row = 0; row < rows; row++)
                {
                    result[row, newCol] = tensor[row, origCol];
                }
            }

            return (TInput)(object)result;
        }
        else
        {
            throw new InvalidOperationException($"Unsupported input type: {input?.GetType().Name ?? "null"}");
        }
    }

    /// <summary>
    /// Selects specific features from a matrix of input data.
    /// </summary>
    /// <param name="X">The original input matrix where rows represent samples and columns represent features.</param>
    /// <param name="selectedFeatures">A list of indices representing the features to be selected.</param>
    /// <returns>A new matrix containing only the selected features.</returns>
    /// <remarks>
    /// This method creates a new matrix with the same number of rows as the input, but only
    /// includes the columns (features) specified in the selectedFeatures list.
    /// </remarks>
    private static Matrix<T> SelectFeaturesMatrix(Matrix<T> X, List<int> selectedFeatures)
    {
        // Validate that all feature indices are within bounds
        foreach (var featureIndex in selectedFeatures)
        {
            if (featureIndex < 0 || featureIndex >= X.Columns)
            {
                throw new ArgumentException($"Invalid feature index: {featureIndex}. Matrix has {X.Columns} columns.");
            }
        }

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
    /// Selects specific features from a tensor of input data.
    /// </summary>
    /// <param name="X">The original input tensor. The first dimension is assumed to be the batch size, 
    /// and the second dimension represents features.</param>
    /// <param name="selectedFeatures">A list of indices representing the features to be selected.</param>
    /// <returns>A new tensor containing only the selected features.</returns>
    /// <remarks>
    /// This method creates a new tensor with the same shape as the input, except for the feature
    /// dimension which is reduced to match the number of selected features. It preserves the
    /// order of samples and any additional dimensions beyond the feature dimension.
    /// </remarks>
    private static Tensor<T> SelectFeaturesTensor(Tensor<T> X, List<int> selectedFeatures)
    {
        if (X.Shape.Length < 2)
        {
            throw new ArgumentException("Input tensor must have at least 2 dimensions", nameof(X));
        }

        // Create a new shape with the updated feature dimension
        var newShape = X.Shape.ToArray();
        newShape[1] = selectedFeatures.Count;

        var selectedX = new Tensor<T>(newShape);

        // Calculate the total number of elements to process
        int totalElements = X.Shape.Aggregate(1, (acc, dim) => acc * dim);
        int featuresCount = X.Shape[1];
        int elementsPerSample = totalElements / X.Shape[0];

        var oldIndices = new int[X.Shape.Length];
        var newIndices = new int[newShape.Length];

        for (int i = 0; i < X.Shape[0]; i++)
        {
            oldIndices[0] = i;
            newIndices[0] = i;
            for (int j = 0; j < selectedFeatures.Count; j++)
            {
                int featureIndex = selectedFeatures[j];
                oldIndices[1] = featureIndex;
                newIndices[1] = j;
                for (int k = 2; k < X.Shape.Length; k++)
                {
                    for (int l = 0; l < X.Shape[k]; l++)
                    {
                        oldIndices[k] = l;
                        newIndices[k] = l;
                        selectedX[newIndices] = X[oldIndices];
                    }
                }
            }
        }

        return selectedX;
    }

    /// <summary>
    /// Determines which column index the feature vector corresponds to in the original matrix.
    /// </summary>
    /// <param name="featureVector">A column vector from the original matrix.</param>
    /// <returns>The index of the column this vector represents in the original matrix.</returns>
    private static int GetFeatureIndex(Vector<T> featureVector, Matrix<T> originalMatrix)
    {
        // We need the original matrix to compare against
        for (int colIndex = 0; colIndex < originalMatrix.Columns; colIndex++)
        {
            var columnVector = originalMatrix.GetColumn(colIndex);
            if (VectorsEqual(columnVector, featureVector))
            {
                return colIndex;
            }
        }

        throw new ArgumentException("Feature vector does not match any column in the original matrix.");
    }

    private static bool VectorsEqual(Vector<T> v1, Vector<T> v2)
    {
        if (v1.Length != v2.Length)
            return false;

        for (int i = 0; i < v1.Length; i++)
        {
            if (!_numOps.Equals(v1[i], v2[i]))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Creates a data container for optimization algorithms with training, validation, and test datasets.
    /// </summary>
    /// <param name="xTrain">Feature input for training the model.</param>
    /// <param name="yTrain">Target values for training the model.</param>
    /// <param name="xValidation">Feature input for validating the model during training.</param>
    /// <param name="yValidation">Target values for validation.</param>
    /// <param name="xTest">Feature input for final testing of the model.</param>
    /// <param name="yTest">Target values for testing.</param>
    /// <returns>A structured container with all datasets needed for optimization.</returns>
    public static OptimizationInputData<T, TInput, TOutput> CreateOptimizationInputData(
        TInput xTrain, TOutput yTrain,
        TInput xValidation, TOutput yValidation,
        TInput xTest, TOutput yTest,
        TInput xFull, TOutput yFull)
    {
        return new OptimizationInputData<T, TInput, TOutput>
        {
            XTrain = xTrain,
            YTrain = yTrain,
            XValidation = xValidation,
            YValidation = yValidation,
            XTest = xTest,
            YTest = yTest,
            XFull = xFull,
            YFull = yFull
        };
    }
}