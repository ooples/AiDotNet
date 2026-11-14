using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing;

/// <summary>
/// Provides simple, standalone utility methods for splitting data into training, validation, and test sets.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Before training a machine learning model, you need to split your data:
/// - Training set: The data the model learns from (typically 70-80%)
/// - Test set: The data used to evaluate how well the model performs on unseen data (typically 20-30%)
/// - Validation set (optional): Data used to tune hyperparameters during training (typically 10-15%)
///
/// This class provides simple static methods to perform these splits without needing to configure
/// a full data preprocessor.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
public static class TrainTestSplit<T>
{
    /// <summary>
    /// Splits data into training and test sets.
    /// </summary>
    /// <param name="X">The feature matrix where each row is a sample.</param>
    /// <param name="y">The target vector containing the values to predict.</param>
    /// <param name="testSize">The proportion of data to use for testing (0.0 to 1.0). Default is 0.2 (20%).</param>
    /// <param name="shuffle">Whether to shuffle the data before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    /// <returns>A tuple containing (XTrain, XTest, yTrain, yTest).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the most common data splitting operation.
    ///
    /// Example usage:
    /// <code>
    /// var (XTrain, XTest, yTrain, yTest) = TrainTestSplit&lt;double&gt;.Split(features, labels, testSize: 0.2);
    /// // XTrain and yTrain: 80% of data for training
    /// // XTest and yTest: 20% of data for testing
    /// </code>
    ///
    /// Why shuffle? If your data is ordered (e.g., by date or category), the split might accidentally
    /// put all similar data in one set. Shuffling ensures a random mix in both sets.
    /// </para>
    /// </remarks>
    public static (Matrix<T> XTrain, Matrix<T> XTest, Vector<T> yTrain, Vector<T> yTest) Split(
        Matrix<T> X,
        Vector<T> y,
        double testSize = 0.2,
        bool shuffle = true,
        int randomSeed = 42)
    {
        ValidateInputs(X, y, testSize);

        int totalSamples = X.Rows;
        int testCount = TestCount(totalSamples, testSize);
        int trainCount = totalSamples - testCount;

        var indices = GetIndices(totalSamples, shuffle, randomSeed);
        var trainRows = indices.Take(trainCount).ToArray();
        var testRows = indices.Skip(trainCount).ToArray();

        return (PreprocessingTensorOps.SelectRows(X, trainRows), PreprocessingTensorOps.SelectRows(X, testRows),
            PreprocessingTensorOps.SelectElements(y, trainRows), PreprocessingTensorOps.SelectElements(y, testRows));
    }

    /// <summary>
    /// scikit-learn's <c>train_test_split</c> test-set size: <c>ceil(test_size * n_samples)</c>; an empty train
    /// set is rejected.
    /// </summary>
    private static int TestCount(int totalSamples, double testSize)
    {
        int testCount = (int)Math.Ceiling(testSize * totalSamples);
        if (totalSamples > 0 && testCount >= totalSamples)
        {
            throw new ArgumentException(
                $"With {totalSamples} samples and testSize={testSize}, the resulting train set would be empty.",
                nameof(testSize));
        }

        return testCount;
    }

    /// <summary>
    /// Splits data into training, validation, and test sets.
    /// </summary>
    /// <param name="X">The feature matrix where each row is a sample.</param>
    /// <param name="y">The target vector containing the values to predict.</param>
    /// <param name="trainSize">The proportion of data to use for training (0.0 to 1.0). Default is 0.7 (70%).</param>
    /// <param name="validationSize">The proportion of data to use for validation (0.0 to 1.0). Default is 0.15 (15%).</param>
    /// <param name="shuffle">Whether to shuffle the data before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    /// <returns>A tuple containing (XTrain, XValidation, XTest, yTrain, yValidation, yTest).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Three-way splits are useful when you need to tune hyperparameters.
    ///
    /// - Training set: The model learns from this data
    /// - Validation set: Used to tune model settings (hyperparameters) without touching test data
    /// - Test set: Final evaluation on completely unseen data
    ///
    /// Example usage:
    /// <code>
    /// var (XTrain, XVal, XTest, yTrain, yVal, yTest) = TrainTestSplit&lt;double&gt;.SplitThreeWay(
    ///     features, labels, trainSize: 0.7, validationSize: 0.15);
    /// // 70% training, 15% validation, 15% test
    /// </code>
    /// </para>
    /// </remarks>
    public static (Matrix<T> XTrain, Matrix<T> XValidation, Matrix<T> XTest,
                   Vector<T> yTrain, Vector<T> yValidation, Vector<T> yTest) SplitThreeWay(
        Matrix<T> X,
        Vector<T> y,
        double trainSize = 0.7,
        double validationSize = 0.15,
        bool shuffle = true,
        int randomSeed = 42)
    {
        ValidateInputsThreeWay(X, y, trainSize, validationSize);

        int totalSamples = X.Rows;
        int trainCount = (int)(totalSamples * trainSize);
        int validationCount = (int)(totalSamples * validationSize);

        var indices = GetIndices(totalSamples, shuffle, randomSeed);
        var trainRows = indices.Take(trainCount).ToArray();
        var validationRows = indices.Skip(trainCount).Take(validationCount).ToArray();
        var testRows = indices.Skip(trainCount + validationCount).ToArray();

        return (PreprocessingTensorOps.SelectRows(X, trainRows), PreprocessingTensorOps.SelectRows(X, validationRows),
            PreprocessingTensorOps.SelectRows(X, testRows), PreprocessingTensorOps.SelectElements(y, trainRows),
            PreprocessingTensorOps.SelectElements(y, validationRows), PreprocessingTensorOps.SelectElements(y, testRows));
    }

    /// <summary>
    /// Splits feature matrix only (without targets) into training and test sets.
    /// </summary>
    /// <param name="X">The feature matrix where each row is a sample.</param>
    /// <param name="testSize">The proportion of data to use for testing (0.0 to 1.0). Default is 0.2 (20%).</param>
    /// <param name="shuffle">Whether to shuffle the data before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    /// <returns>A tuple containing (XTrain, XTest).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you only have features and no target labels,
    /// which is common in unsupervised learning tasks like clustering.
    /// </para>
    /// </remarks>
    public static (Matrix<T> XTrain, Matrix<T> XTest) SplitX(
        Matrix<T> X,
        double testSize = 0.2,
        bool shuffle = true,
        int randomSeed = 42)
    {
        if (X is null)
        {
            throw new ArgumentNullException(nameof(X));
        }

        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentException("testSize must be between 0 and 1 (exclusive).", nameof(testSize));
        }

        int totalSamples = X.Rows;
        int testCount = TestCount(totalSamples, testSize);
        int trainCount = totalSamples - testCount;

        var indices = GetIndices(totalSamples, shuffle, randomSeed);
        return (PreprocessingTensorOps.SelectRows(X, indices.Take(trainCount).ToArray()),
            PreprocessingTensorOps.SelectRows(X, indices.Skip(trainCount).ToArray()));
    }

    /// <summary>
    /// Splits data into k folds for cross-validation.
    /// </summary>
    /// <param name="X">The feature matrix where each row is a sample.</param>
    /// <param name="y">The target vector containing the values to predict.</param>
    /// <param name="k">The number of folds. Default is 5.</param>
    /// <param name="shuffle">Whether to shuffle the data before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    /// <returns>A list of tuples, each containing (XTrain, XTest, yTrain, yTest) for that fold.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> K-fold cross-validation splits data into k equal parts.
    /// Each part takes turns being the test set while the others form the training set.
    /// This gives you k different train/test splits to evaluate your model.
    ///
    /// Example with k=5:
    /// - Fold 1: Parts 2,3,4,5 train, Part 1 test
    /// - Fold 2: Parts 1,3,4,5 train, Part 2 test
    /// - ...and so on
    ///
    /// This helps you get a more reliable estimate of model performance.
    /// </para>
    /// </remarks>
    public static List<(Matrix<T> XTrain, Matrix<T> XTest, Vector<T> yTrain, Vector<T> yTest)> KFoldSplit(
        Matrix<T> X,
        Vector<T> y,
        int k = 5,
        bool shuffle = true,
        int randomSeed = 42)
    {
        if (X is null)
        {
            throw new ArgumentNullException(nameof(X));
        }

        if (y is null)
        {
            throw new ArgumentNullException(nameof(y));
        }

        if (X.Rows != y.Length)
        {
            throw new ArgumentException("X and y must have the same number of samples.");
        }

        if (k < 2)
        {
            throw new ArgumentException("k must be at least 2.", nameof(k));
        }

        if (k > X.Rows)
        {
            throw new ArgumentException("k cannot be greater than the number of samples.", nameof(k));
        }

        int totalSamples = X.Rows;
        var indices = GetIndices(totalSamples, shuffle, randomSeed);

        // scikit-learn KFold: every fold has n / k samples and the first n % k folds one more
        var folds = new List<(Matrix<T>, Matrix<T>, Vector<T>, Vector<T>)>();
        int testStart = 0;
        for (int fold = 0; fold < k; fold++)
        {
            int testCount = totalSamples / k + (fold < totalSamples % k ? 1 : 0);
            int testEnd = testStart + testCount;
            var testRows = indices.Skip(testStart).Take(testCount).ToArray();
            var trainRows = indices.Take(testStart).Concat(indices.Skip(testEnd)).ToArray();

            folds.Add((PreprocessingTensorOps.SelectRows(X, trainRows), PreprocessingTensorOps.SelectRows(X, testRows),
                PreprocessingTensorOps.SelectElements(y, trainRows), PreprocessingTensorOps.SelectElements(y, testRows)));
            testStart = testEnd;
        }

        return folds;
    }

    private static int[] GetIndices(int totalSamples, bool shuffle, int randomSeed)
    {
        var indices = Enumerable.Range(0, totalSamples).ToArray();

        if (shuffle)
        {
            var random = RandomHelper.CreateSeededRandom(randomSeed);
            // Fisher-Yates shuffle
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        return indices;
    }

    private static void ValidateInputs(Matrix<T> X, Vector<T> y, double testSize)
    {
        if (X is null)
        {
            throw new ArgumentNullException(nameof(X));
        }

        if (y is null)
        {
            throw new ArgumentNullException(nameof(y));
        }

        if (X.Rows != y.Length)
        {
            throw new ArgumentException("X and y must have the same number of samples.");
        }

        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentException("testSize must be between 0 and 1 (exclusive).", nameof(testSize));
        }
    }

    private static void ValidateInputsThreeWay(Matrix<T> X, Vector<T> y, double trainSize, double validationSize)
    {
        if (X is null)
        {
            throw new ArgumentNullException(nameof(X));
        }

        if (y is null)
        {
            throw new ArgumentNullException(nameof(y));
        }

        if (X.Rows != y.Length)
        {
            throw new ArgumentException("X and y must have the same number of samples.");
        }

        if (trainSize <= 0 || trainSize >= 1)
        {
            throw new ArgumentException("trainSize must be between 0 and 1 (exclusive).", nameof(trainSize));
        }

        if (validationSize <= 0 || validationSize >= 1)
        {
            throw new ArgumentException("validationSize must be between 0 and 1 (exclusive).", nameof(validationSize));
        }

        if (trainSize + validationSize >= 1)
        {
            throw new ArgumentException("trainSize + validationSize must be less than 1.");
        }
    }
}
