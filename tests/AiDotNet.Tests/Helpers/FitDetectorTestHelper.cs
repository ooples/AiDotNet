using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Statistics;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Helper class for creating test data for FitDetector tests.
/// </summary>
public static class FitDetectorTestHelper
{
    /// <summary>
    /// Creates ErrorStats from actual and predicted vectors.
    /// Uses internal constructors accessible via InternalsVisibleTo.
    /// </summary>
    public static ErrorStats<double> CreateErrorStats(Vector<double> actual, Vector<double> predicted, int featureCount = 2)
    {
        var inputs = new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = featureCount,
            PredictionType = PredictionType.Regression
        };
        return new ErrorStats<double>(inputs);
    }

    /// <summary>
    /// Creates PredictionStats from actual and predicted vectors.
    /// Uses internal constructors accessible via InternalsVisibleTo.
    /// </summary>
    public static PredictionStats<double> CreatePredictionStats(Vector<double> actual, Vector<double> predicted, int numberOfParameters = 2)
    {
        var inputs = new PredictionStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            NumberOfParameters = numberOfParameters,
            ConfidenceLevel = 0.95,
            LearningCurveSteps = 10,
            PredictionType = PredictionType.Regression
        };
        return new PredictionStats<double>(inputs);
    }

    /// <summary>
    /// Creates ModelStats from actual, predicted, and feature data.
    /// Uses internal constructors accessible via InternalsVisibleTo.
    /// </summary>
    public static ModelStats<double, Matrix<double>, Vector<double>> CreateModelStats(
        Vector<double> actual,
        Vector<double> predicted,
        Matrix<double>? features = null)
    {
        var featureMatrix = features ?? new Matrix<double>(actual.Length, 2);
        var inputs = new ModelStatsInputs<double, Matrix<double>, Vector<double>>
        {
            Actual = actual,
            Predicted = predicted,
            XMatrix = featureMatrix,
            FeatureCount = featureMatrix.Columns
        };
        return new ModelStats<double, Matrix<double>, Vector<double>>(inputs);
    }

    /// <summary>
    /// Creates a complete ModelEvaluationData with proper statistics.
    /// </summary>
    public static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateEvaluationData(
        Vector<double> trainActual, Vector<double> trainPredicted,
        Vector<double>? validationActual = null, Vector<double>? validationPredicted = null,
        Vector<double>? testActual = null, Vector<double>? testPredicted = null,
        Matrix<double>? features = null)
    {
        var evalData = new ModelEvaluationData<double, Matrix<double>, Vector<double>>();

        // Set training set stats
        evalData.TrainingSet.ErrorStats = CreateErrorStats(trainActual, trainPredicted);
        evalData.TrainingSet.PredictionStats = CreatePredictionStats(trainActual, trainPredicted);
        evalData.TrainingSet.Actual = trainActual;
        evalData.TrainingSet.Predicted = trainPredicted;

        // Set validation set stats if provided
        if (validationActual is not null && validationPredicted is not null)
        {
            evalData.ValidationSet.ErrorStats = CreateErrorStats(validationActual, validationPredicted);
            evalData.ValidationSet.PredictionStats = CreatePredictionStats(validationActual, validationPredicted);
            evalData.ValidationSet.Actual = validationActual;
            evalData.ValidationSet.Predicted = validationPredicted;
        }
        else
        {
            // Use training data as default
            evalData.ValidationSet.ErrorStats = CreateErrorStats(trainActual, trainPredicted);
            evalData.ValidationSet.PredictionStats = CreatePredictionStats(trainActual, trainPredicted);
            evalData.ValidationSet.Actual = trainActual;
            evalData.ValidationSet.Predicted = trainPredicted;
        }

        // Set test set stats if provided
        if (testActual is not null && testPredicted is not null)
        {
            evalData.TestSet.ErrorStats = CreateErrorStats(testActual, testPredicted);
            evalData.TestSet.PredictionStats = CreatePredictionStats(testActual, testPredicted);
            evalData.TestSet.Actual = testActual;
            evalData.TestSet.Predicted = testPredicted;
        }
        else
        {
            // Use training data as default
            evalData.TestSet.ErrorStats = CreateErrorStats(trainActual, trainPredicted);
            evalData.TestSet.PredictionStats = CreatePredictionStats(trainActual, trainPredicted);
            evalData.TestSet.Actual = trainActual;
            evalData.TestSet.Predicted = trainPredicted;
        }

        // Set model stats - use provided features or create appropriate default matrix
        var featureMatrix = features ?? CreateFeatureMatrix(trainActual.Length, 3);
        evalData.ModelStats = CreateModelStats(trainActual, trainPredicted, featureMatrix);

        return evalData;
    }

    /// <summary>
    /// Creates vectors that will result in approximately the target MSE.
    /// MSE = mean((actual - predicted)^2)
    /// Creates continuous data with variance, including both positive and non-positive values
    /// to satisfy both AUC calculation and confidence interval requirements.
    /// </summary>
    public static (Vector<double> Actual, Vector<double> Predicted) CreateVectorsWithTargetMse(double targetMse, int length = 30)
    {
        // Ensure minimum length for statistical calculations
        // Must match the minimum feature matrix rows for consistency
        if (length < 20)
        {
            length = 30;
        }

        var actual = new double[length];
        var predicted = new double[length];
        var random = RandomHelper.CreateSeededRandom(42); // Fixed seed for reproducibility
        // For uniform distribution on [-a, a], variance = a²/3
        // Since error = (random - 0.5) * errorScale * 2 ranges from -errorScale to +errorScale
        // The actual variance = errorScale² / 3
        // To get MSE = targetMse, we need errorScale² / 3 = targetMse, so errorScale = sqrt(3 * targetMse)
        var errorScale = Math.Sqrt(targetMse * 3);

        // Create continuous data with variance that includes:
        // - Some negative/zero values for AUC "negative" class
        // - Some positive values for AUC "positive" class
        // - Sufficient variance for Weibull confidence intervals
        for (int i = 0; i < length; i++)
        {
            // Create actual values ranging from -1 to 2 (includes both positive and non-positive)
            // First third are negative/zero, rest are positive
            if (i < length / 3)
            {
                // Non-positive values (for AUC negative class)
                actual[i] = -0.5 + random.NextDouble() * 0.5; // Range: -0.5 to 0.0
            }
            else
            {
                // Positive values (for AUC positive class)
                actual[i] = 0.5 + random.NextDouble() * 1.5; // Range: 0.5 to 2.0
            }

            // Create predicted values with controlled error
            predicted[i] = actual[i] + (random.NextDouble() - 0.5) * errorScale * 2;
        }

        return (new Vector<double>(actual), new Vector<double>(predicted));
    }

    /// <summary>
    /// Creates a well-conditioned feature matrix for testing.
    /// Uses random data with different seeds per column to ensure non-singular correlation matrix.
    /// Note: For SimpleRegression tests (HeteroscedasticityFitDetector), pass columns=1.
    /// For VIF tests, use columns >= 2.
    /// </summary>
    public static Matrix<double> CreateFeatureMatrix(int rows, int columns = 3)
    {
        // Ensure minimum dimensions for proper statistics calculation
        if (rows < 20) rows = 30;
        // Only enforce minimum of 2 columns if caller didn't explicitly request fewer
        // (SimpleRegression requires exactly 1 column)
        if (columns < 1) columns = 3;

        var data = new double[rows, columns];

        // Use different seeds for each column to ensure truly uncorrelated features
        for (int j = 0; j < columns; j++)
        {
            var random = RandomHelper.CreateSeededRandom(42 + j * 1000); // Different seed per column
            for (int i = 0; i < rows; i++)
            {
                // Use purely random values in different ranges per column
                // This ensures columns are completely independent
                data[i, j] = random.NextDouble() * 10.0 + (j + 1) * 5.0;
            }
        }

        return new Matrix<double>(data);
    }
}
