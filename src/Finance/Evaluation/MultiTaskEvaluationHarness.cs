using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;

namespace AiDotNet.Finance.Evaluation;

/// <summary>
/// Multi-task evaluation harness for time series foundation models, supporting standardized
/// evaluation across forecasting, anomaly detection, classification, imputation, and embedding tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Foundation models like MOMENT support multiple tasks with one model.
/// This harness lets you evaluate how well a model performs across all its supported tasks
/// using standardized metrics for each:
/// <list type="bullet">
/// <item><b>Forecasting:</b> MSE, MAE, RMSE, MASE</item>
/// <item><b>Anomaly Detection:</b> Precision, Recall, F1-Score</item>
/// <item><b>Classification:</b> Accuracy, F1-Score</item>
/// <item><b>Imputation:</b> MSE on imputed values</item>
/// <item><b>Embedding:</b> Silhouette score for clustering quality</item>
/// </list>
/// </para>
/// </remarks>
public class MultiTaskEvaluationHarness<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    private static T NaN => NumOps.FromDouble(double.NaN);

    /// <summary>The first <paramref name="count"/> elements of <paramref name="tensor"/> in row-major order, as a rank-1 tensor.</summary>
    private static Tensor<T> FlatPrefix(Tensor<T> tensor, int count)
    {
        var flat = Engine.Reshape(tensor, new[] { tensor.Length });
        return count == tensor.Length ? flat : Engine.TensorSlice(flat, new[] { 0 }, new[] { count });
    }

    /// <summary>The sum of per-series scalars (zero when there are none).</summary>
    private static T SumOf(List<T> values)
        => values.Count == 0 ? NumOps.Zero : Engine.TensorSum(new Tensor<T>(new[] { values.Count }, new Vector<T>(values.ToArray())));

    /// <summary>
    /// Evaluates a foundation model's forecasting performance.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="testPairs">List of (context, target) tensor pairs.</param>
    /// <returns>Dictionary of metric names to values.</returns>
    public Dictionary<string, T> EvaluateForecasting(
        ITimeSeriesFoundationModel<T> model,
        IReadOnlyList<(Tensor<T> Context, Tensor<T> Target)> testPairs)
    {
        Guard.NotNull(model);
        Guard.NotNull(testPairs);

        T totalMse = NumOps.Zero, totalMae = NumOps.Zero;
        int totalPoints = 0;

        foreach (var (context, target) in testPairs)
        {
            var prediction = model.Forecast(context, null);
            int evalLen = Math.Min(prediction.Length, target.Length);

            if (evalLen <= 0)
                continue;

            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            var diff = Engine.TensorSubtract(FlatPrefix(prediction, evalLen), FlatPrefix(target, evalLen));
            totalMse = NumOps.Add(totalMse, Engine.TensorSumOfSquares(diff));
            totalMae = NumOps.Add(totalMae, Engine.TensorSum(Engine.TensorAbs(diff)));
            totalPoints += evalLen;
        }

        if (totalPoints == 0)
            return new Dictionary<string, T> { ["MSE"] = NaN, ["MAE"] = NaN };

        T mse = NumOps.Divide(totalMse, NumOps.FromDouble(totalPoints));
        T mae = NumOps.Divide(totalMae, NumOps.FromDouble(totalPoints));

        return new Dictionary<string, T>
        {
            ["MSE"] = mse,
            ["MAE"] = mae,
            ["RMSE"] = NumOps.Sqrt(mse),
            ["NumSamples"] = NumOps.FromDouble(testPairs.Count),
            ["NumPoints"] = NumOps.FromDouble(totalPoints)
        };
    }

    /// <summary>
    /// Evaluates a foundation model's anomaly detection performance.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="testSeries">List of time series to test.</param>
    /// <param name="groundTruthLabels">Binary labels (1 = anomaly, 0 = normal) per series per timestep.</param>
    /// <param name="threshold">Anomaly score threshold (if null, uses model default).</param>
    /// <returns>Dictionary of metric names to values.</returns>
    public Dictionary<string, T> EvaluateAnomalyDetection(
        ITimeSeriesFoundationModel<T> model,
        IReadOnlyList<Tensor<T>> testSeries,
        IReadOnlyList<Tensor<T>> groundTruthLabels,
        double? threshold = null)
    {
        Guard.NotNull(model);
        Guard.NotNull(testSeries);
        Guard.NotNull(groundTruthLabels);

        if (!model.SupportedTasks.Contains(TimeSeriesFoundationModelTask.AnomalyDetection))
        {
            return new Dictionary<string, T>
            {
                ["Supported"] = NumOps.Zero,
                ["Precision"] = NaN,
                ["Recall"] = NaN,
                ["F1"] = NaN
            };
        }

        if (testSeries.Count != groundTruthLabels.Count)
            throw new ArgumentException(
                $"testSeries count ({testSeries.Count}) must match groundTruthLabels count ({groundTruthLabels.Count}).",
                nameof(groundTruthLabels));

        // Per-series counts, summed once after the loop.
        var tpCounts = new List<T>();
        var predictedCounts = new List<T>();
        var actualCounts = new List<T>();
        T binarizationThreshold = NumOps.FromDouble(threshold ?? 0.5);
        T labelThreshold = NumOps.FromDouble(0.5);

        for (int s = 0; s < testSeries.Count; s++)
        {
            // Pass threshold through to the model (null lets the model use its own default)
            var scores = model.DetectAnomalies(testSeries[s], threshold);
            var labels = groundTruthLabels[s];

            // For binarization: use the explicit threshold if provided,
            // otherwise use 0.5 to binarize the model's returned scores (hoisted above the loop).
            int evalLen = Math.Min(scores.Length, labels.Length);

            if (evalLen <= 0)
                continue;

            // 1 where the score clears the threshold / the label is an anomaly, else 0.
            var predicted = Engine.TensorGreaterThan(FlatPrefix(scores, evalLen), binarizationThreshold);
            var actual = Engine.TensorGreaterThan(FlatPrefix(labels, evalLen), labelThreshold);
            tpCounts.Add(Engine.TensorSum(Engine.TensorMultiply(predicted, actual)));
            predictedCounts.Add(Engine.TensorSum(predicted));
            actualCounts.Add(Engine.TensorSum(actual));
        }

        T tp = SumOf(tpCounts), predictedPositives = SumOf(predictedCounts), actualPositives = SumOf(actualCounts);

        T fp = NumOps.Subtract(predictedPositives, tp);
        T fn = NumOps.Subtract(actualPositives, tp);
        T precision = NumOps.GreaterThan(predictedPositives, NumOps.Zero) ? NumOps.Divide(tp, predictedPositives) : NumOps.Zero;
        T recall = NumOps.GreaterThan(actualPositives, NumOps.Zero) ? NumOps.Divide(tp, actualPositives) : NumOps.Zero;
        T precisionPlusRecall = NumOps.Add(precision, recall);
        T f1 = NumOps.GreaterThan(precisionPlusRecall, NumOps.Zero)
            ? NumOps.Divide(NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(precision, recall)), precisionPlusRecall)
            : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["Supported"] = NumOps.One,
            ["Precision"] = precision,
            ["Recall"] = recall,
            ["F1"] = f1,
            ["TruePositives"] = tp,
            ["FalsePositives"] = fp,
            ["FalseNegatives"] = fn
        };
    }

    /// <summary>
    /// Evaluates a foundation model's classification performance.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="testSeries">List of time series to classify.</param>
    /// <param name="trueLabels">Ground truth class labels (0-indexed).</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <returns>Dictionary of metric names to values.</returns>
    public Dictionary<string, T> EvaluateClassification(
        ITimeSeriesFoundationModel<T> model,
        IReadOnlyList<Tensor<T>> testSeries,
        IReadOnlyList<int> trueLabels,
        int numClasses)
    {
        Guard.NotNull(model);
        Guard.NotNull(testSeries);
        Guard.NotNull(trueLabels);

        if (!model.SupportedTasks.Contains(TimeSeriesFoundationModelTask.Classification))
        {
            return new Dictionary<string, T>
            {
                ["Supported"] = NumOps.Zero,
                ["Accuracy"] = NaN
            };
        }

        int correct = 0;
        int total = 0;

        for (int s = 0; s < testSeries.Count && s < trueLabels.Count; s++)
        {
            var logits = model.Classify(testSeries[s], numClasses);

            // Find argmax (first maximum among the first numClasses logits)
            int predictedClass = 0;
            int classCount = Math.Min(logits.Length, numClasses);
            if (classCount > 0)
                predictedClass = Engine.TensorArgMax(FlatPrefix(logits, classCount), 0).AsSpan()[0];

            if (predictedClass == trueLabels[s])
                correct++;
            total++;
        }

        T accuracy = total > 0 ? NumOps.Divide(NumOps.FromDouble(correct), NumOps.FromDouble(total)) : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["Supported"] = NumOps.One,
            ["Accuracy"] = accuracy,
            ["Correct"] = NumOps.FromDouble(correct),
            ["Total"] = NumOps.FromDouble(total)
        };
    }

    /// <summary>
    /// Evaluates a foundation model's imputation performance.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="completeSeries">Complete (no missing values) time series for evaluation.</param>
    /// <param name="masks">Binary masks indicating which values to "remove" and impute (0 = masked).</param>
    /// <returns>Dictionary of metric names to values.</returns>
    public Dictionary<string, T> EvaluateImputation(
        ITimeSeriesFoundationModel<T> model,
        IReadOnlyList<Tensor<T>> completeSeries,
        IReadOnlyList<Tensor<T>> masks)
    {
        Guard.NotNull(model);
        Guard.NotNull(completeSeries);
        Guard.NotNull(masks);

        if (!model.SupportedTasks.Contains(TimeSeriesFoundationModelTask.Imputation))
        {
            return new Dictionary<string, T>
            {
                ["Supported"] = NumOps.Zero,
                ["MSE"] = NaN
            };
        }

        T totalMse = NumOps.Zero;
        T totalMasked = NumOps.Zero;

        if (completeSeries.Count != masks.Count)
            throw new ArgumentException(
                $"completeSeries count ({completeSeries.Count}) must match masks count ({masks.Count}).",
                nameof(masks));

        for (int s = 0; s < completeSeries.Count; s++)
        {
            var original = completeSeries[s];
            var mask = masks[s];

            if (original.Length != mask.Length)
                throw new ArgumentException(
                    $"Series {s} length ({original.Length}) must match mask length ({mask.Length}).",
                    nameof(masks));

            // Create masked input (zero out masked positions)
            var flatOriginal = FlatPrefix(original, original.Length);
            var keep = Engine.TensorGreaterThan(FlatPrefix(mask, mask.Length), NumOps.FromDouble(0.5));
            var maskedInput = Engine.Reshape(
                Engine.TensorWhere(keep, flatOriginal, Engine.TensorZerosLike(flatOriginal)),
                (int[])original._shape.Clone());

            var imputed = model.Impute(maskedInput, mask);

            // Evaluate only on masked positions
            int evalLen = Math.Min(imputed.Length, original.Length);
            if (evalLen > 0)
            {
                // Squared error at the positions that were masked (mask < 0.5); the rest contribute nothing.
                var wasMasked = Engine.TensorLessThan(FlatPrefix(mask, evalLen), NumOps.FromDouble(0.5));
                var diff = Engine.TensorSubtract(FlatPrefix(imputed, evalLen), FlatPrefix(original, evalLen));
                var squared = Engine.TensorMultiply(diff, diff);
                totalMse = NumOps.Add(totalMse, Engine.TensorSum(Engine.TensorWhere(wasMasked, squared, Engine.TensorZerosLike(squared))));
                totalMasked = NumOps.Add(totalMasked, Engine.TensorSum(wasMasked));
            }
        }

        T mse = NumOps.GreaterThan(totalMasked, NumOps.Zero) ? NumOps.Divide(totalMse, totalMasked) : NaN;

        return new Dictionary<string, T>
        {
            ["Supported"] = NumOps.One,
            ["MSE"] = mse,
            ["RMSE"] = NumOps.Sqrt(mse),
            ["NumMaskedPoints"] = totalMasked
        };
    }

    /// <summary>
    /// Runs a full multi-task evaluation across all supported tasks.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="forecastingData">Optional forecasting test data.</param>
    /// <param name="anomalyData">Optional anomaly detection test data.</param>
    /// <param name="classificationData">Optional classification test data.</param>
    /// <param name="imputationData">Optional imputation test data.</param>
    /// <returns>Nested dictionary of task name to metric name to value.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Pass data for whichever tasks you want to evaluate. Tasks without
    /// data will be skipped. The model's SupportedTasks will also be checked — you'll get
    /// a "Supported: 0" indicator for tasks the model doesn't handle.
    /// </remarks>
    public Dictionary<string, Dictionary<string, T>> RunFullEvaluation(
        ITimeSeriesFoundationModel<T> model,
        IReadOnlyList<(Tensor<T> Context, Tensor<T> Target)>? forecastingData = null,
        (IReadOnlyList<Tensor<T>> Series, IReadOnlyList<Tensor<T>> Labels, double? Threshold)? anomalyData = null,
        (IReadOnlyList<Tensor<T>> Series, IReadOnlyList<int> Labels, int NumClasses)? classificationData = null,
        (IReadOnlyList<Tensor<T>> Series, IReadOnlyList<Tensor<T>> Masks)? imputationData = null)
    {
        Guard.NotNull(model);

        var results = new Dictionary<string, Dictionary<string, T>>();

        // Model info
        results["ModelInfo"] = new Dictionary<string, T>
        {
            ["SupportedTaskCount"] = NumOps.FromDouble(model.SupportedTasks.Count),
            ["MaxContextLength"] = NumOps.FromDouble(model.MaxContextLength),
            ["MaxPredictionHorizon"] = NumOps.FromDouble(model.MaxPredictionHorizon)
        };

        // Forecasting
        if (forecastingData is not null)
            results["Forecasting"] = EvaluateForecasting(model, forecastingData);

        // Anomaly Detection
        if (anomalyData.HasValue)
            results["AnomalyDetection"] = EvaluateAnomalyDetection(
                model, anomalyData.Value.Series, anomalyData.Value.Labels, anomalyData.Value.Threshold);

        // Classification
        if (classificationData.HasValue)
            results["Classification"] = EvaluateClassification(
                model, classificationData.Value.Series, classificationData.Value.Labels,
                classificationData.Value.NumClasses);

        // Imputation
        if (imputationData.HasValue)
            results["Imputation"] = EvaluateImputation(
                model, imputationData.Value.Series, imputationData.Value.Masks);

        return results;
    }
}
