using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Preprocessing;

namespace AiDotNet.Models.Results;

/// <summary>
/// Partial class providing model evaluation functionality through the AiModelResult facade.
/// </summary>
public partial class AiModelResult<T, TInput, TOutput>
{
    private readonly PredictionStatsOptions _evaluationPredictionOptions = new();

    /// <summary>
    /// Evaluates the model across training, validation, and test datasets.
    /// </summary>
    /// <param name="inputData">The optimization input data containing training, validation, and test splits.</param>
    /// <param name="predictionTypeOverride">Optional override for the prediction type. If not specified, will be inferred from the data.</param>
    /// <returns>Comprehensive evaluation data including statistics for all datasets.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method evaluates how well your model performs across different datasets:
    /// - Training data: The data your model learned from
    /// - Validation data: Data used to fine-tune your model during training
    /// - Test data: Completely new data to see how well your model generalizes
    ///
    /// The returned evaluation data includes error statistics, prediction quality metrics, and model statistics
    /// that help you understand your model's strengths and weaknesses.
    /// </para>
    /// </remarks>
    public ModelEvaluationData<T, TInput, TOutput> EvaluateFull(
        OptimizationInputData<T, TInput, TOutput> inputData,
        PredictionType? predictionTypeOverride = null)
    {
        var inferredPredictionType = predictionTypeOverride
            ?? PredictionTypeInference.InferFromTargets<T, TOutput>(inputData.YTrain);

        var trainingSet = CalculateDataSetStatsInternal(
            inputData.XTrain,
            inputData.YTrain,
            inferredPredictionType);

        // Only calculate validation stats if validation data is provided and has content
        DataSetStats<T, TInput, TOutput>? validationSet = null;
        if (inputData.XValidation != null && InputHelper<T, TInput>.GetInputSize(inputData.XValidation) > 0)
        {
            validationSet = CalculateDataSetStatsInternal(
                inputData.XValidation,
                inputData.YValidation,
                inferredPredictionType);
        }

        // Only calculate test stats if test data is provided and has content
        DataSetStats<T, TInput, TOutput>? testSet = null;
        if (inputData.XTest != null && InputHelper<T, TInput>.GetInputSize(inputData.XTest) > 0)
        {
            testSet = CalculateDataSetStatsInternal(
                inputData.XTest,
                inputData.YTest,
                inferredPredictionType);
        }

        // ModelStats uses validation data if available, otherwise training data
        var statsForModelCalc = validationSet ?? trainingSet;

        var evaluationData = new ModelEvaluationData<T, TInput, TOutput>
        {
            TrainingSet = trainingSet,
            ValidationSet = validationSet ?? new DataSetStats<T, TInput, TOutput>(),
            TestSet = testSet ?? new DataSetStats<T, TInput, TOutput>(),
            ModelStats = TryCalculateModelStatsInternal(
                statsForModelCalc.Features,
                statsForModelCalc.Actual,
                statsForModelCalc.Predicted)
        };

        return evaluationData;
    }

    /// <summary>
    /// Gets evaluation statistics for a single dataset.
    /// </summary>
    /// <param name="X">The input features.</param>
    /// <param name="y">The target values.</param>
    /// <param name="predictionType">Optional prediction type. If not specified, will be inferred from targets.</param>
    /// <returns>Comprehensive statistics about the model's performance on this dataset.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method evaluates your model on a single dataset and returns:
    /// - Error statistics (MSE, RMSE, MAE, etc.)
    /// - Basic statistics for actual and predicted values
    /// - Prediction quality metrics (R-squared, adjusted R-squared, etc.)
    ///
    /// Use this when you want to evaluate on just one dataset rather than the full train/val/test split.
    /// </para>
    /// </remarks>
    public DataSetStats<T, TInput, TOutput> GetDataSetStats(
        TInput X,
        TOutput y,
        PredictionType? predictionType = null)
    {
        var inferredPredictionType = predictionType
            ?? PredictionTypeInference.InferFromTargets<T, TOutput>(y);

        return CalculateDataSetStatsInternal(X, y, inferredPredictionType);
    }

    /// <summary>
    /// Internal method to calculate dataset statistics.
    /// </summary>
    private DataSetStats<T, TInput, TOutput> CalculateDataSetStatsInternal(
        TInput X,
        TOutput y,
        PredictionType predictionType)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Cannot evaluate a null model. Ensure the model is initialized.");
        }

        var predictions = Model.Predict(X);
        var inputSize = InputHelper<T, TInput>.GetInputSize(X);

        if (!TryGetAlignedVectorsInternal(y, predictions, predictionType, out var actual, out var predicted))
        {
            var emptyStats = new DataSetStats<T, TInput, TOutput>
            {
                ErrorStats = ErrorStats<T>.Empty(),
                ActualBasicStats = BasicStats<T>.Empty(),
                PredictedBasicStats = BasicStats<T>.Empty(),
                PredictionStats = PredictionStats<T>.Empty(),
                Predicted = predictions,
                Features = X,
                Actual = y,
                IsDataProvided = true
            };

            TryPopulateUncertaintyStatsInternal(emptyStats, X);
            return emptyStats;
        }

        var stats = new DataSetStats<T, TInput, TOutput>
        {
            ErrorStats = CalculateErrorStatsInternal(actual, predicted, inputSize, predictionType),
            ActualBasicStats = CalculateBasicStatsInternal(actual),
            PredictedBasicStats = CalculateBasicStatsInternal(predicted),
            PredictionStats = CalculatePredictionStatsInternal(actual, predicted, inputSize, predictionType),
            Predicted = predictions,
            Features = X,
            Actual = y,
            IsDataProvided = true
        };

        TryPopulateUncertaintyStatsInternal(stats, X);
        return stats;
    }

    private void TryPopulateUncertaintyStatsInternal(DataSetStats<T, TInput, TOutput> stats, TInput X)
    {
        if (UncertaintyQuantificationOptions is not { Enabled: true })
        {
            return;
        }

        var uq = PredictWithUncertainty(X);
        var numOps = MathHelper.GetNumericOperations<T>();

        stats.UncertaintyStats = new UncertaintyStats<T>();
        if (uq.Metrics.TryGetValue("predictive_entropy", out var predictiveEntropy))
        {
            stats.UncertaintyStats.Metrics["predictive_entropy"] = MeanOfTensor(predictiveEntropy, numOps);
        }

        if (uq.Metrics.TryGetValue("mutual_information", out var mutualInformation))
        {
            stats.UncertaintyStats.Metrics["mutual_information"] = MeanOfTensor(mutualInformation, numOps);
        }

        if (HasExpectedCalibrationError)
        {
            stats.UncertaintyStats.Metrics["expected_calibration_error"] = ExpectedCalibrationError;
        }
    }

    private static T MeanOfTensor(Tensor<T> values, INumericOperations<T> numOps)
    {
        if (values.Length == 0)
        {
            return numOps.Zero;
        }

        var sum = numOps.Zero;
        for (int i = 0; i < values.Length; i++)
        {
            sum = numOps.Add(sum, values[i]);
        }

        return numOps.Divide(sum, numOps.FromDouble(values.Length));
    }

    private static bool TryGetAlignedVectorsInternal(
        TOutput actualOutput,
        TOutput predictedOutput,
        PredictionType predictionType,
        out Vector<T> actual,
        out Vector<T> predicted)
    {
        actual = Vector<T>.Empty();
        predicted = Vector<T>.Empty();

        bool preferMultiClassMatrixPath = predictionType == PredictionType.MultiClass
            && (LooksLikeMultiClassScoresInternal(actualOutput) || LooksLikeMultiClassScoresInternal(predictedOutput));

        try
        {
            actual = ConversionsHelper.ConvertToVector<T, TOutput>(actualOutput);
            predicted = ConversionsHelper.ConvertToVector<T, TOutput>(predictedOutput);

            if (!preferMultiClassMatrixPath && actual.Length == predicted.Length)
            {
                return true;
            }
        }
        catch (InvalidOperationException) { }
        catch (ArgumentException) { }
        catch (NotSupportedException) { }

        if (predictionType == PredictionType.MultiClass)
        {
            if (TryGetMultiClassLabelVectorsInternal(actualOutput, predictedOutput, ref actual, ref predicted))
            {
                return true;
            }
        }

        if (TryGetFlattenedMatrixVectorsInternal(actualOutput, predictedOutput, out actual, out predicted))
        {
            return true;
        }

        return false;
    }

    private static bool LooksLikeMultiClassScoresInternal(TOutput output)
    {
        if (output is Matrix<T> matrix)
        {
            return matrix.Columns > 1;
        }

        if (output is Tensor<T> tensor)
        {
            return tensor.Rank == 2 && tensor.Shape.Length >= 2 && tensor.Shape[1] > 1;
        }

        return false;
    }

    private static bool TryGetMultiClassLabelVectorsInternal(
        TOutput actualOutput,
        TOutput predictedOutput,
        ref Vector<T> actual,
        ref Vector<T> predicted)
    {
        try
        {
            var predictedMatrix = ConversionsHelper.ConvertToMatrix<T, TOutput>(predictedOutput);
            if (predictedMatrix.Rows <= 0 || predictedMatrix.Columns <= 0)
            {
                return false;
            }

            Vector<T> actualLabels;
            if (actual.Length > 0)
            {
                actualLabels = actual;
            }
            else
            {
                actualLabels = ConversionsHelper.ConvertToVector<T, TOutput>(actualOutput);
            }

            if (actualLabels.Length == predictedMatrix.Rows)
            {
                actual = actualLabels;
                predicted = ArgMaxToLabelVectorInternal(predictedMatrix);
                return true;
            }
        }
        catch (InvalidOperationException) { }
        catch (ArgumentException) { }
        catch (NotSupportedException) { }

        try
        {
            var actualMatrix = ConversionsHelper.ConvertToMatrix<T, TOutput>(actualOutput);
            var predictedMatrix = ConversionsHelper.ConvertToMatrix<T, TOutput>(predictedOutput);

            if (actualMatrix.Rows != predictedMatrix.Rows || actualMatrix.Columns != predictedMatrix.Columns)
            {
                return false;
            }

            if (actualMatrix.Rows <= 0 || actualMatrix.Columns <= 0)
            {
                return false;
            }

            actual = ArgMaxToLabelVectorInternal(actualMatrix);
            predicted = ArgMaxToLabelVectorInternal(predictedMatrix);
            return actual.Length == predicted.Length;
        }
        catch (InvalidOperationException) { }
        catch (ArgumentException) { }
        catch (NotSupportedException) { }

        return false;
    }

    private static bool TryGetFlattenedMatrixVectorsInternal(
        TOutput actualOutput,
        TOutput predictedOutput,
        out Vector<T> actual,
        out Vector<T> predicted)
    {
        actual = Vector<T>.Empty();
        predicted = Vector<T>.Empty();

        try
        {
            var actualMatrix = ConversionsHelper.ConvertToMatrix<T, TOutput>(actualOutput);
            var predictedMatrix = ConversionsHelper.ConvertToMatrix<T, TOutput>(predictedOutput);

            if (actualMatrix.Rows != predictedMatrix.Rows || actualMatrix.Columns != predictedMatrix.Columns)
            {
                return false;
            }

            if (actualMatrix.Rows <= 0 || actualMatrix.Columns <= 0)
            {
                return false;
            }

            actual = FlattenToVectorInternal(actualMatrix);
            predicted = FlattenToVectorInternal(predictedMatrix);
            return actual.Length == predicted.Length;
        }
        catch (InvalidOperationException) { }
        catch (ArgumentException) { }
        catch (NotSupportedException) { }

        return false;
    }

    private static Vector<T> ArgMaxToLabelVectorInternal(Matrix<T> scores)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var labels = new Vector<T>(scores.Rows);

        for (int row = 0; row < scores.Rows; row++)
        {
            int bestIndex = 0;
            T bestValue = scores[row, 0];

            for (int col = 1; col < scores.Columns; col++)
            {
                var value = scores[row, col];
                if (numOps.GreaterThan(value, bestValue))
                {
                    bestValue = value;
                    bestIndex = col;
                }
            }

            labels[row] = numOps.FromDouble(bestIndex);
        }

        return labels;
    }

    private static Vector<T> FlattenToVectorInternal(Matrix<T> matrix)
    {
        var flattened = new Vector<T>(matrix.Rows * matrix.Columns);
        int index = 0;

        for (int row = 0; row < matrix.Rows; row++)
        {
            for (int col = 0; col < matrix.Columns; col++)
            {
                flattened[index++] = matrix[row, col];
            }
        }

        return flattened;
    }

    private static ErrorStats<T> CalculateErrorStatsInternal(
        Vector<T> actual,
        Vector<T> predicted,
        int featureCount,
        PredictionType predictionType)
    {
        return new ErrorStats<T>(new ErrorStatsInputs<T>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = featureCount,
            PredictionType = predictionType
        });
    }

    private static BasicStats<T> CalculateBasicStatsInternal(Vector<T> values)
    {
        return new BasicStats<T>(new BasicStatsInputs<T> { Values = values });
    }

    private PredictionStats<T> CalculatePredictionStatsInternal(
        Vector<T> actual,
        Vector<T> predicted,
        int featureCount,
        PredictionType predictionType)
    {
        return new PredictionStats<T>(new PredictionStatsInputs<T>
        {
            Actual = actual,
            Predicted = predicted,
            NumberOfParameters = featureCount,
            ConfidenceLevel = _evaluationPredictionOptions.ConfidenceLevel,
            LearningCurveSteps = _evaluationPredictionOptions.LearningCurveSteps,
            PredictionType = predictionType
        });
    }

    private ModelStats<T, TInput, TOutput> TryCalculateModelStatsInternal(
        TInput xForStatistics,
        TOutput actual,
        TOutput predicted)
    {
        try
        {
            return CalculateModelStatsInternal(xForStatistics, actual, predicted);
        }
        catch (InvalidOperationException) { return ModelStats<T, TInput, TOutput>.Empty(); }
        catch (ArgumentException) { return ModelStats<T, TInput, TOutput>.Empty(); }
        catch (NotSupportedException) { return ModelStats<T, TInput, TOutput>.Empty(); }
        catch (ArithmeticException) { return ModelStats<T, TInput, TOutput>.Empty(); }
        catch (IndexOutOfRangeException) { return ModelStats<T, TInput, TOutput>.Empty(); }
    }

    private ModelStats<T, TInput, TOutput> CalculateModelStatsInternal(
        TInput xForStatistics,
        TOutput actual,
        TOutput predicted)
    {
        return new ModelStats<T, TInput, TOutput>(new ModelStatsInputs<T, TInput, TOutput>
        {
            XMatrix = xForStatistics,
            FeatureCount = InputHelper<T, TInput>.GetInputSize(xForStatistics),
            Actual = actual,
            Predicted = predicted,
            Model = Model
        });
    }
}
