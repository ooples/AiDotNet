using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet;

public partial class PredictionModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Configures uncertainty quantification (UQ) for inference-time uncertainty estimates.
    /// </summary>
    /// <param name="options">
    /// Optional uncertainty quantification options. When null, industry-standard defaults are used and UQ is enabled.
    /// </param>
    /// <param name="calibrationData">
    /// Optional calibration data used by conformal prediction and probability calibration methods. When null, calibration-dependent
    /// features are skipped unless the library can infer calibration behavior from other configuration.
    /// </param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Uncertainty quantification augments point predictions with uncertainty signals such as variance, predictive entropy, and mutual information.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This feature lets you ask the model not only "what is the prediction?", but also "how sure are you?".
    /// </para>
    /// <para>
    /// Typical usage:
    /// <code>
    /// var loader = DataLoaders.FromTensors(xTrain, yTrain);
    /// var result = await new PredictionModelBuilder&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;()
    ///     .ConfigureDataLoader(loader)
    ///     .ConfigureModel(model)
    ///     .ConfigureOptimizer(optimizer)
    ///     .ConfigureUncertaintyQuantification()
    ///     .BuildAsync();
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureUncertaintyQuantification(
        UncertaintyQuantificationOptions? options = null,
        UncertaintyCalibrationData<TInput, TOutput>? calibrationData = null)
    {
        _uncertaintyQuantificationOptions = options ?? new UncertaintyQuantificationOptions { Enabled = true };
        _uncertaintyCalibrationData = calibrationData;
        return this;
    }

    /// <summary>
    /// Computes calibration artifacts (not raw calibration data) and attaches them to the final result.
    /// </summary>
    /// <param name="result">The prediction result to attach artifacts to.</param>
    /// <remarks>
    /// <para>
    /// This method is internal to the builder and does not expand the public API surface.
    /// It computes artifacts such as conformal thresholds or temperature scaling parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Calibration makes uncertainty estimates more reliable by tuning them on a small held-out dataset.
    /// </para>
    /// </remarks>
    private void TryComputeAndAttachUncertaintyCalibrationArtifacts(PredictionModelResult<T, TInput, TOutput> result)
    {
        if (_uncertaintyQuantificationOptions is not { Enabled: true })
        {
            return;
        }

        var effectiveMethod = _uncertaintyQuantificationOptions.Method;
        if (effectiveMethod != UncertaintyQuantificationMethod.Auto &&
            effectiveMethod != UncertaintyQuantificationMethod.ConformalPrediction)
        {
            return;
        }

        var artifacts = new UncertaintyCalibrationArtifacts<T>();
        var calibrationData = _uncertaintyCalibrationData;
        if (calibrationData is { HasLabels: true } && calibrationData.Labels is { } labels)
        {
            TryComputeClassificationCalibrationArtifacts(result, calibrationData.X, labels, _uncertaintyQuantificationOptions, artifacts);
        }
        else if (calibrationData is { HasTargets: true })
        {
            TryComputeRegressionConformalArtifacts(result, calibrationData.X, calibrationData.Y, artifacts);
        }

        if (artifacts.HasConformalRegression || artifacts.HasConformalClassification || artifacts.HasTemperatureScaling)
        {
            result.SetUncertaintyCalibrationArtifacts(artifacts);
        }
    }

    /// <summary>
    /// Computes conformal regression artifacts from calibration data.
    /// </summary>
    /// <param name="result">The trained model result.</param>
    /// <param name="xCalibration">Calibration inputs.</param>
    /// <param name="yCalibration">Calibration targets.</param>
    /// <param name="artifacts">Artifact container to populate.</param>
    private static void TryComputeRegressionConformalArtifacts(
        PredictionModelResult<T, TInput, TOutput> result,
        TInput xCalibration,
        TOutput yCalibration,
        UncertaintyCalibrationArtifacts<T> artifacts)
    {
        var actual = ConversionsHelper.ConvertToVector<T, TOutput>(yCalibration);
        var predictedOutput = result.Predict(xCalibration);
        var predicted = ConversionsHelper.ConvertToVector<T, TOutput>(predictedOutput);

        if (actual.Length == 0 || actual.Length != predicted.Length)
        {
            return;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var residuals = new Vector<T>(actual.Length);
        for (int i = 0; i < actual.Length; i++)
        {
            residuals[i] = numOps.Abs(numOps.Subtract(actual[i], predicted[i]));
        }

        var quantile = ComputeConformalRegressionQuantile(residuals, result.UncertaintyQuantificationOptions?.ConformalConfidenceLevel ?? 0.9);

        artifacts.HasConformalRegression = true;
        artifacts.ConformalRegressionQuantile = quantile;
    }

    /// <summary>
    /// Computes the conformal regression residual quantile for a desired confidence level.
    /// </summary>
    /// <param name="residuals">Residuals (order does not matter).</param>
    /// <param name="confidenceLevel">Desired coverage probability (e.g., 0.9).</param>
    /// <returns>The residual quantile used as the prediction interval half-width.</returns>
    private static T ComputeConformalRegressionQuantile(Vector<T> residuals, double confidenceLevel)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var n = residuals.Length;
        if (n == 0)
        {
            return numOps.Zero;
        }

        var adjustedLevel = Math.Ceiling((n + 1) * confidenceLevel) / n;
        if (adjustedLevel > 1.0)
        {
            adjustedLevel = 1.0;
        }

        var index = (int)Math.Ceiling(n * adjustedLevel) - 1;
        if (index < 0) index = 0;
        if (index >= n) index = n - 1;

        return SelectKthInPlace(residuals, index);
    }

    /// <summary>
    /// Computes classification calibration artifacts including ECE, conformal threshold, and optional temperature scaling.
    /// </summary>
    /// <param name="result">The trained model result.</param>
    /// <param name="xCalibration">Calibration inputs.</param>
    /// <param name="labels">True class labels for calibration samples.</param>
    /// <param name="options">Uncertainty quantification options.</param>
    /// <param name="artifacts">Artifact container to populate.</param>
    private static void TryComputeClassificationCalibrationArtifacts(
        PredictionModelResult<T, TInput, TOutput> result,
        TInput xCalibration,
        Vector<int> labels,
        UncertaintyQuantificationOptions options,
        UncertaintyCalibrationArtifacts<T> artifacts)
    {
        if (labels.Length == 0)
        {
            return;
        }

        var probsOutput = result.Predict(xCalibration);
        var probsTensor = ConversionsHelper.ConvertToTensor<T>(probsOutput!).Clone();
        if (probsTensor.Rank < 1)
        {
            return;
        }

        var numClasses = probsTensor.Shape[probsTensor.Shape.Length - 1];
        if (numClasses <= 1)
        {
            return;
        }

        var batch = probsTensor.Rank == 1 ? 1 : probsTensor.Length / numClasses;
        if (batch <= 0)
        {
            return;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var temperature = options.EnableTemperatureScaling
            ? FitTemperatureFromProbabilities(probsTensor, labels, batch, numClasses)
            : numOps.One;

        if (options.EnableTemperatureScaling)
        {
            artifacts.HasTemperatureScaling = true;
            artifacts.TemperatureScalingTemperature = temperature;
        }

        var calibrated = options.EnableTemperatureScaling
            ? ApplyTemperatureScalingToProbabilities(probsTensor, temperature, batch, numClasses)
            : probsTensor;

        var flatCalibrated = calibrated.ToVector();
        var predictions = new Vector<int>(batch);
        var confidence = new Vector<T>(batch);
        for (int i = 0; i < batch; i++)
        {
            var bestLabel = 0;
            var bestScore = flatCalibrated[i * numClasses];
            for (int c = 1; c < numClasses; c++)
            {
                var score = flatCalibrated[i * numClasses + c];
                if (numOps.GreaterThan(score, bestScore))
                {
                    bestScore = score;
                    bestLabel = c;
                }
            }

            predictions[i] = bestLabel;
            confidence[i] = bestScore;
        }

        var ece = new AiDotNet.UncertaintyQuantification.Calibration.ExpectedCalibrationError<T>(numBins: 10);
        artifacts.HasExpectedCalibrationError = true;
        artifacts.ExpectedCalibrationError = ece.Compute(confidence, predictions, labels);

        var scores = new Vector<T>(batch);
        var validCount = 0;
        for (int i = 0; i < batch; i++)
        {
            var label = labels[i];
            if (label < 0 || label >= numClasses)
            {
                System.Diagnostics.Debug.WriteLine(
                    $"Warning: Conformal classification skipped an invalid label at index {i}: {label}. Valid range is [0, {numClasses - 1}].");
                continue;
            }

            scores[validCount++] = flatCalibrated[i * numClasses + label];
        }

        if (validCount == 0)
        {
            System.Diagnostics.Debug.WriteLine(
                "Warning: Conformal classification threshold could not be computed because no valid labels were provided.");
            return;
        }

        var validScores = validCount == scores.Length
            ? scores
            : scores.Subvector(0, validCount);

        var threshold = ComputeConformalClassificationThreshold(validScores, result.UncertaintyQuantificationOptions?.ConformalConfidenceLevel ?? 0.9);

        artifacts.HasConformalClassification = true;
        artifacts.ConformalClassificationThreshold = threshold;
        artifacts.ConformalClassificationNumClasses = numClasses;
    }

    /// <summary>
    /// Fits a temperature scaling value using log-probabilities derived from predicted probabilities.
    /// </summary>
    /// <param name="probabilities">Predicted probabilities.</param>
    /// <param name="labels">True labels.</param>
    /// <param name="batch">Number of samples.</param>
    /// <param name="classes">Number of classes.</param>
    /// <returns>The fitted temperature parameter.</returns>
    private static T FitTemperatureFromProbabilities(Tensor<T> probabilities, Vector<int> labels, int batch, int classes)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var eps = numOps.FromDouble(1e-12);
        var sumTolerance = numOps.FromDouble(1e-3);
        var uniformProbability = numOps.Divide(numOps.One, numOps.FromDouble(classes));
        var logits = new Matrix<T>(batch, classes);
        var flat = probabilities.ToVector();

        for (int i = 0; i < batch; i++)
        {
            var sum = numOps.Zero;
            for (int c = 0; c < classes; c++)
            {
                var p = flat[i * classes + c];
                if (numOps.LessThan(p, numOps.Zero))
                {
                    p = numOps.Zero;
                }
                sum = numOps.Add(sum, p);
            }

            var fallbackToUniform = numOps.LessThan(sum, eps);
            var shouldNormalize = !fallbackToUniform && numOps.GreaterThan(numOps.Abs(numOps.Subtract(sum, numOps.One)), sumTolerance);

            for (int c = 0; c < classes; c++)
            {
                var p = fallbackToUniform ? uniformProbability : flat[i * classes + c];
                if (!fallbackToUniform)
                {
                    if (numOps.LessThan(p, numOps.Zero))
                    {
                        p = numOps.Zero;
                    }
                    if (shouldNormalize)
                    {
                        p = numOps.Divide(p, sum);
                    }
                }
                if (numOps.LessThan(p, eps))
                {
                    p = eps;
                }
                logits[i, c] = numOps.Log(p);
            }
        }

        var scaler = new AiDotNet.UncertaintyQuantification.Calibration.TemperatureScaling<T>(initialTemperature: 1.0);
        scaler.Calibrate(logits, labels);
        return scaler.Temperature;
    }

    /// <summary>
    /// Applies temperature scaling to probabilities and renormalizes to maintain a valid distribution.
    /// </summary>
    /// <param name="probabilities">Input probabilities.</param>
    /// <param name="temperature">Temperature parameter.</param>
    /// <param name="batch">Number of samples.</param>
    /// <param name="classes">Number of classes.</param>
    /// <returns>Temperature-scaled probabilities.</returns>
    private static Tensor<T> ApplyTemperatureScalingToProbabilities(
        Tensor<T> probabilities,
        T temperature,
        int batch,
        int classes)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var eps = numOps.FromDouble(1e-12);
        var scaled = new Vector<T>(batch * classes);
        var flat = probabilities.ToVector();

        for (int i = 0; i < batch; i++)
        {
            var maxLogit = default(T)!;
            for (int c = 0; c < classes; c++)
            {
                var p = flat[i * classes + c];
                if (numOps.LessThan(p, eps))
                {
                    p = eps;
                }
                var logit = numOps.Divide(numOps.Log(p), temperature);
                if (c == 0 || numOps.GreaterThan(logit, maxLogit))
                {
                    maxLogit = logit;
                }
            }

            var sumExp = numOps.Zero;
            for (int c = 0; c < classes; c++)
            {
                var p = flat[i * classes + c];
                if (numOps.LessThan(p, eps))
                {
                    p = eps;
                }
                var logit = numOps.Divide(numOps.Log(p), temperature);
                var exp = numOps.Exp(numOps.Subtract(logit, maxLogit));
                scaled[i * classes + c] = exp;
                sumExp = numOps.Add(sumExp, exp);
            }

            for (int c = 0; c < classes; c++)
            {
                scaled[i * classes + c] = numOps.Divide(scaled[i * classes + c], sumExp);
            }
        }

        return new Tensor<T>([batch, classes], scaled).Reshape(probabilities.Shape);
    }

    /// <summary>
    /// Computes the conformal score threshold for classification prediction sets.
    /// </summary>
    /// <param name="scores">Scores for the true labels (order does not matter).</param>
    /// <param name="confidenceLevel">Desired coverage probability (e.g., 0.9).</param>
    /// <returns>The score threshold.</returns>
    private static T ComputeConformalClassificationThreshold(Vector<T> scores, double confidenceLevel)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var n = scores.Length;
        if (n == 0)
        {
            return numOps.Zero;
        }

        var adjustedLevel = Math.Ceiling((n + 1) * (1.0 - confidenceLevel)) / n;
        if (adjustedLevel < 0.0)
        {
            adjustedLevel = 0.0;
        }
        if (adjustedLevel > 1.0)
        {
            adjustedLevel = 1.0;
        }

        var index = (int)Math.Floor(n * adjustedLevel);
        if (index < 0) index = 0;
        if (index >= n) index = n - 1;

        return SelectKthInPlace(scores, index);
    }

    /// <summary>
    /// Selects the k-th smallest element in-place using a Quickselect partitioning strategy.
    /// </summary>
    /// <param name="values">Values to select from (will be reordered).</param>
    /// <param name="k">Zero-based rank of the element to select.</param>
    /// <returns>The k-th smallest element.</returns>
    private static T SelectKthInPlace(Vector<T> values, int k)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var span = values.AsWritableSpan();
        var left = 0;
        var right = span.Length - 1;

        int Partition(Span<T> innerSpan, int innerLeft, int innerRight, int pivotIndex)
        {
            var pivotValue = innerSpan[pivotIndex];
            Swap(innerSpan, pivotIndex, innerRight);

            var storeIndex = innerLeft;
            for (int i = innerLeft; i < innerRight; i++)
            {
                if (numOps.LessThan(innerSpan[i], pivotValue))
                {
                    Swap(innerSpan, storeIndex, i);
                    storeIndex++;
                }
            }

            Swap(innerSpan, innerRight, storeIndex);
            return storeIndex;
        }

        while (true)
        {
            if (left == right)
            {
                return span[left];
            }

            var pivotIndex = left + ((right - left) / 2);
            pivotIndex = Partition(span, left, right, pivotIndex);

            if (k == pivotIndex)
            {
                return span[k];
            }

            if (k < pivotIndex)
            {
                right = pivotIndex - 1;
            }
            else
            {
                left = pivotIndex + 1;
            }
        }
    }

    /// <summary>
    /// Swaps two elements in a span.
    /// </summary>
    /// <param name="span">The span containing the elements to swap.</param>
    /// <param name="a">The index of the first element.</param>
    /// <param name="b">The index of the second element.</param>
    private static void Swap(Span<T> span, int a, int b)
    {
        if (a == b)
        {
            return;
        }

        var temp = span[a];
        span[a] = span[b];
        span[b] = temp;
    }
}
