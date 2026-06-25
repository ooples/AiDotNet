using AiDotNet.Clustering.Evaluation;
using AiDotNet.Clustering.Interfaces;
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

    private ModelEvaluationData<T, TInput, TOutput>? _cachedEvaluation;

    /// <summary>
    /// Gets the trained model's evaluation metrics, computed once from the data the model was built on and cached.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the single, model-type-aware home for every metric the build captured. It adapts to what you trained:
    /// supervised models populate <see cref="ModelEvaluationData{T, TInput, TOutput}.TrainingSet"/>,
    /// <see cref="ModelEvaluationData{T, TInput, TOutput}.ValidationSet"/> and
    /// <see cref="ModelEvaluationData{T, TInput, TOutput}.TestSet"/>; clustering models populate
    /// <see cref="ModelEvaluationData{T, TInput, TOutput}.ClusteringMetrics"/>; other model types report their
    /// specialised scores under <see cref="ModelEvaluationData{T, TInput, TOutput}.AdditionalMetrics"/>.
    /// </para>
    /// <para><b>For Beginners:</b> After you call <c>BuildAsync()</c>, this is the one place to read how well your
    /// model did — you do not need to re-pass your data. For example:
    /// <list type="bullet">
    /// <item><description>Regression: <c>result.Evaluation.TestSet.PredictionStats.R2</c>,
    /// <c>result.Evaluation.TestSet.ErrorStats.RMSE</c></description></item>
    /// <item><description>Classification: <c>result.Evaluation.TestSet.ErrorStats.Accuracy</c>,
    /// <c>result.Evaluation.TestSet.ErrorStats.F1Score</c></description></item>
    /// <item><description>Clustering: <c>result.Evaluation.ClusteringMetrics.Silhouette</c></description></item>
    /// </list>
    /// The value is computed lazily on first access from the statistics captured during the build, so reading it is
    /// cheap and never re-runs the model. To evaluate on a brand-new dataset instead, call
    /// <see cref="GetDataSetStats"/> or <see cref="EvaluateFull"/> with that data.
    /// </para>
    /// </remarks>
    public ModelEvaluationData<T, TInput, TOutput> Evaluation =>
        _cachedEvaluation ??= BuildCachedEvaluationInternal();

    private ModelEvaluationData<T, TInput, TOutput> BuildCachedEvaluationInternal()
    {
        var evaluation = new ModelEvaluationData<T, TInput, TOutput>();

        var optimizationResult = OptimizationResult;
        if (optimizationResult is not null)
        {
            evaluation.TrainingSet = ToDataSetStatsInternal(optimizationResult.TrainingResult);
            evaluation.ValidationSet = ToDataSetStatsInternal(optimizationResult.ValidationResult);
            evaluation.TestSet = ToDataSetStatsInternal(optimizationResult.TestResult);
        }

        TryPopulateClusteringMetricsInternal(evaluation);

        return evaluation;
    }

    /// <summary>
    /// Maps an optimization dataset result (already computed during the build) onto the richer
    /// <see cref="DataSetStats{T, TInput, TOutput}"/> surface exposed through <see cref="Evaluation"/>.
    /// </summary>
    private static DataSetStats<T, TInput, TOutput> ToDataSetStatsInternal(
        OptimizationResult<T, TInput, TOutput>.DatasetResult datasetResult)
    {
        if (datasetResult is null)
        {
            return new DataSetStats<T, TInput, TOutput>();
        }

        bool hasData = (object?)datasetResult.X != null
            && InputHelper<T, TInput>.GetInputSize(datasetResult.X) > 0;

        return new DataSetStats<T, TInput, TOutput>
        {
            ErrorStats = datasetResult.ErrorStats ?? ErrorStats<T>.Empty(),
            PredictionStats = datasetResult.PredictionStats ?? PredictionStats<T>.Empty(),
            ActualBasicStats = datasetResult.ActualBasicStats ?? BasicStats<T>.Empty(),
            PredictedBasicStats = datasetResult.PredictedBasicStats ?? BasicStats<T>.Empty(),
            Predicted = datasetResult.Predictions,
            Features = datasetResult.X,
            Actual = datasetResult.Y,
            IsDataProvided = hasData
        };
    }

    /// <summary>
    /// Computes internal clustering quality metrics for unsupervised clustering models, so that
    /// <see cref="Evaluation"/> reports cluster-appropriate scores instead of (meaningless) supervised error stats.
    /// </summary>
    private void TryPopulateClusteringMetricsInternal(ModelEvaluationData<T, TInput, TOutput> evaluation)
    {
        if (Model is not IClustering<T> clusteringModel)
        {
            return;
        }

        var labels = clusteringModel.Labels;
        if (labels is null || labels.Length == 0)
        {
            return;
        }

        // Internal clustering metrics compare each point to its assigned cluster, so the data matrix and the label
        // vector must describe the same points. The training split holds the data the model was fit on.
        var trainingResult = OptimizationResult?.TrainingResult;
        if (trainingResult is null)
        {
            return;
        }

        // Box through object first: a type pattern on the unconstrained TInput is not allowed directly.
        object? trainingFeatures = trainingResult.X;
        if (trainingFeatures is not Matrix<T> trainingData
            || trainingData.Rows == 0
            || labels.Length != trainingData.Rows)
        {
            return;
        }

        try
        {
            evaluation.ClusteringMetrics = new ClusterMetrics<T>().Evaluate(trainingData, labels);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or ArithmeticException)
        {
            // Clustering metrics are best-effort; leave ClusteringMetrics null but leave a breadcrumb
            // so a downstream NRE on result.Evaluation.ClusteringMetrics is traceable.
            System.Diagnostics.Trace.TraceWarning(
                $"Clustering metrics could not be computed: {ex.GetType().Name}: {ex.Message}");
        }
    }

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
