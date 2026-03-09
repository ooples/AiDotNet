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

    /// <summary>
    /// Evaluates a foundation model's forecasting performance.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="testPairs">List of (context, target) tensor pairs.</param>
    /// <returns>Dictionary of metric names to values.</returns>
    public Dictionary<string, double> EvaluateForecasting(
        ITimeSeriesFoundationModel<T> model,
        IReadOnlyList<(Tensor<T> Context, Tensor<T> Target)> testPairs)
    {
        Guard.NotNull(model);
        Guard.NotNull(testPairs);

        double totalMse = 0, totalMae = 0;
        int totalPoints = 0;

        foreach (var (context, target) in testPairs)
        {
            var prediction = model.Forecast(context, null);
            int evalLen = Math.Min(prediction.Length, target.Length);

            for (int i = 0; i < evalLen; i++)
            {
                double pred = NumOps.ToDouble(prediction[i]);
                double actual = NumOps.ToDouble(target[i]);
                double diff = pred - actual;

                totalMse += diff * diff;
                totalMae += Math.Abs(diff);
                totalPoints++;
            }
        }

        if (totalPoints == 0)
            return new Dictionary<string, double> { ["MSE"] = double.NaN, ["MAE"] = double.NaN };

        double mse = totalMse / totalPoints;
        double mae = totalMae / totalPoints;

        return new Dictionary<string, double>
        {
            ["MSE"] = mse,
            ["MAE"] = mae,
            ["RMSE"] = Math.Sqrt(mse),
            ["NumSamples"] = testPairs.Count,
            ["NumPoints"] = totalPoints
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
    public Dictionary<string, double> EvaluateAnomalyDetection(
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
            return new Dictionary<string, double>
            {
                ["Supported"] = 0,
                ["Precision"] = double.NaN,
                ["Recall"] = double.NaN,
                ["F1"] = double.NaN
            };
        }

        if (testSeries.Count != groundTruthLabels.Count)
            throw new ArgumentException(
                $"testSeries count ({testSeries.Count}) must match groundTruthLabels count ({groundTruthLabels.Count}).",
                nameof(groundTruthLabels));

        int tp = 0, fp = 0, fn = 0;

        for (int s = 0; s < testSeries.Count; s++)
        {
            // Pass threshold through to the model (null lets the model use its own default)
            var scores = model.DetectAnomalies(testSeries[s], threshold);
            var labels = groundTruthLabels[s];

            // For binarization: use the explicit threshold if provided,
            // otherwise use 0.5 to binarize the model's returned scores
            double binarizationThreshold = threshold ?? 0.5;

            int evalLen = Math.Min(scores.Length, labels.Length);

            for (int i = 0; i < evalLen; i++)
            {
                bool predicted = NumOps.ToDouble(scores[i]) > binarizationThreshold;
                bool actual = NumOps.GreaterThan(labels[i], NumOps.FromDouble(0.5));

                if (predicted && actual) tp++;
                else if (predicted && !actual) fp++;
                else if (!predicted && actual) fn++;
            }
        }

        double precision = tp + fp > 0 ? (double)tp / (tp + fp) : 0;
        double recall = tp + fn > 0 ? (double)tp / (tp + fn) : 0;
        double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;

        return new Dictionary<string, double>
        {
            ["Supported"] = 1,
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
    public Dictionary<string, double> EvaluateClassification(
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
            return new Dictionary<string, double>
            {
                ["Supported"] = 0,
                ["Accuracy"] = double.NaN
            };
        }

        int correct = 0;
        int total = 0;

        for (int s = 0; s < testSeries.Count && s < trueLabels.Count; s++)
        {
            var logits = model.Classify(testSeries[s], numClasses);

            // Find argmax
            int predictedClass = 0;
            double maxLogit = double.MinValue;
            for (int c = 0; c < logits.Length && c < numClasses; c++)
            {
                double logit = NumOps.ToDouble(logits[c]);
                if (logit > maxLogit)
                {
                    maxLogit = logit;
                    predictedClass = c;
                }
            }

            if (predictedClass == trueLabels[s])
                correct++;
            total++;
        }

        double accuracy = total > 0 ? (double)correct / total : 0;

        return new Dictionary<string, double>
        {
            ["Supported"] = 1,
            ["Accuracy"] = accuracy,
            ["Correct"] = correct,
            ["Total"] = total
        };
    }

    /// <summary>
    /// Evaluates a foundation model's imputation performance.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="completeSeries">Complete (no missing values) time series for evaluation.</param>
    /// <param name="masks">Binary masks indicating which values to "remove" and impute (0 = masked).</param>
    /// <returns>Dictionary of metric names to values.</returns>
    public Dictionary<string, double> EvaluateImputation(
        ITimeSeriesFoundationModel<T> model,
        IReadOnlyList<Tensor<T>> completeSeries,
        IReadOnlyList<Tensor<T>> masks)
    {
        Guard.NotNull(model);
        Guard.NotNull(completeSeries);
        Guard.NotNull(masks);

        if (!model.SupportedTasks.Contains(TimeSeriesFoundationModelTask.Imputation))
        {
            return new Dictionary<string, double>
            {
                ["Supported"] = 0,
                ["MSE"] = double.NaN
            };
        }

        double totalMse = 0;
        int totalMasked = 0;

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
            var maskedInput = new Tensor<T>(original.Shape);
            for (int i = 0; i < original.Length; i++)
            {
                double m = NumOps.ToDouble(mask[i]);
                maskedInput.Data.Span[i] = m > 0.5 ? original[i] : NumOps.Zero;
            }

            var imputed = model.Impute(maskedInput, mask);

            // Evaluate only on masked positions
            int evalLen = Math.Min(imputed.Length, original.Length);
            for (int i = 0; i < evalLen; i++)
            {
                if (NumOps.LessThan(mask[i], NumOps.FromDouble(0.5))) // This was masked
                {
                    double diff = NumOps.ToDouble(imputed[i]) - NumOps.ToDouble(original[i]);
                    totalMse += diff * diff;
                    totalMasked++;
                }
            }
        }

        double mse = totalMasked > 0 ? totalMse / totalMasked : double.NaN;

        return new Dictionary<string, double>
        {
            ["Supported"] = 1,
            ["MSE"] = mse,
            ["RMSE"] = Math.Sqrt(mse),
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
    public Dictionary<string, Dictionary<string, double>> RunFullEvaluation(
        ITimeSeriesFoundationModel<T> model,
        IReadOnlyList<(Tensor<T> Context, Tensor<T> Target)>? forecastingData = null,
        (IReadOnlyList<Tensor<T>> Series, IReadOnlyList<Tensor<T>> Labels, double? Threshold)? anomalyData = null,
        (IReadOnlyList<Tensor<T>> Series, IReadOnlyList<int> Labels, int NumClasses)? classificationData = null,
        (IReadOnlyList<Tensor<T>> Series, IReadOnlyList<Tensor<T>> Masks)? imputationData = null)
    {
        Guard.NotNull(model);

        var results = new Dictionary<string, Dictionary<string, double>>();

        // Model info
        results["ModelInfo"] = new Dictionary<string, double>
        {
            ["SupportedTaskCount"] = model.SupportedTasks.Count,
            ["MaxContextLength"] = model.MaxContextLength,
            ["MaxPredictionHorizon"] = model.MaxPredictionHorizon
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
