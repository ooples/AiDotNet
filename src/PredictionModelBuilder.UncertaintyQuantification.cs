using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

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
        _uncertaintyQuantificationOptions.Normalize();
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
            effectiveMethod != UncertaintyQuantificationMethod.ConformalPrediction &&
            effectiveMethod != UncertaintyQuantificationMethod.LaplaceApproximation &&
            effectiveMethod != UncertaintyQuantificationMethod.Swag)
        {
            return;
        }

        var artifacts = new UncertaintyCalibrationArtifacts<T>();
        var calibrationData = _uncertaintyCalibrationData;
        if (calibrationData is { HasLabels: true } && calibrationData.Labels is { } labels)
        {
            TryComputeClassificationCalibrationArtifacts(result, calibrationData.X, labels, _uncertaintyQuantificationOptions, artifacts);

            if (effectiveMethod == UncertaintyQuantificationMethod.LaplaceApproximation)
            {
                TryComputeLaplacePosteriorArtifacts(result, calibrationData.X, labels, _uncertaintyQuantificationOptions, artifacts);
            }
            else if (effectiveMethod == UncertaintyQuantificationMethod.Swag)
            {
                TryComputeSwagPosteriorArtifacts(result, calibrationData.X, labels, _uncertaintyQuantificationOptions, artifacts);
            }
        }
        else if (calibrationData is { HasTargets: true })
        {
            TryComputeRegressionConformalArtifacts(result, calibrationData.X, calibrationData.Y, artifacts);

            if (effectiveMethod == UncertaintyQuantificationMethod.LaplaceApproximation)
            {
                TryComputeLaplacePosteriorArtifacts(result, calibrationData.X, calibrationData.Y, _uncertaintyQuantificationOptions, artifacts);
            }
            else if (effectiveMethod == UncertaintyQuantificationMethod.Swag)
            {
                TryComputeSwagPosteriorArtifacts(result, calibrationData.X, calibrationData.Y, _uncertaintyQuantificationOptions, artifacts);
            }
        }

        if (artifacts.HasConformalRegression ||
            artifacts.HasConformalClassification ||
            artifacts.HasTemperatureScaling ||
            artifacts.HasPlattScaling ||
            artifacts.HasIsotonicRegressionCalibration ||
            artifacts.HasLaplacePosterior ||
            artifacts.HasSwagPosterior)
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

        probsTensor = EnsureProbabilityTensor(probsTensor, batch, numClasses);

        var numOps = MathHelper.GetNumericOperations<T>();
        var effectiveCalibration = ResolveCalibrationMethod(options, numClasses);
        var calibrated = probsTensor;

        if (effectiveCalibration == ProbabilityCalibrationMethod.TemperatureScaling && options.EnableTemperatureScaling)
        {
            var temperature = FitTemperatureFromProbabilities(probsTensor, labels, batch, numClasses);
            artifacts.HasTemperatureScaling = true;
            artifacts.TemperatureScalingTemperature = temperature;
            calibrated = ApplyTemperatureScalingToProbabilities(probsTensor, temperature, batch, numClasses);
        }
        else if (effectiveCalibration == ProbabilityCalibrationMethod.PlattScaling && options.EnablePlattScaling && numClasses == 2)
        {
            var (a, b) = FitPlattScalingBinary(probsTensor, labels, batch);
            artifacts.HasPlattScaling = true;
            var aVec = new Vector<T>(1);
            aVec[0] = a;
            artifacts.PlattScalingA = aVec;
            var bVec = new Vector<T>(1);
            bVec[0] = b;
            artifacts.PlattScalingB = bVec;
            calibrated = ApplyPlattScalingToProbabilitiesBinary(probsTensor, a, b, batch);
        }
        else if (effectiveCalibration == ProbabilityCalibrationMethod.IsotonicRegression &&
                 options.EnableIsotonicRegressionCalibration &&
                 numClasses == 2)
        {
            var (x, y) = FitIsotonicCalibrationBinary(probsTensor, labels, batch);
            artifacts.HasIsotonicRegressionCalibration = true;
            artifacts.IsotonicCalibrationX = x;
            artifacts.IsotonicCalibrationY = y;
            calibrated = ApplyIsotonicCalibrationToProbabilitiesBinary(probsTensor, x, y, batch);
        }

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
        var scoreConfidence = new Vector<T>(batch);
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
            scoreConfidence[validCount - 1] = confidence[i];
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

        var confidenceLevel = result.UncertaintyQuantificationOptions?.ConformalConfidenceLevel ?? 0.9;
        var threshold = ComputeConformalClassificationThreshold(validScores.Clone(), confidenceLevel);

        if (options.ConformalMode == ConformalPredictionMode.CrossConformal)
        {
            threshold = ComputeCrossConformalClassificationThreshold(validScores, confidenceLevel, options.CrossConformalFolds, options.RandomSeed);
        }
        else if (options.ConformalMode == ConformalPredictionMode.Adaptive)
        {
            var validConf = validCount == scoreConfidence.Length
                ? scoreConfidence
                : scoreConfidence.Subvector(0, validCount);

            var (edges, thresholds) = ComputeAdaptiveConformalClassificationThresholds(
                validScores,
                validConf,
                confidenceLevel,
                options.AdaptiveConformalBins);

            artifacts.HasAdaptiveConformalClassification = true;
            artifacts.ConformalClassificationAdaptiveBinEdges = edges;
            artifacts.ConformalClassificationAdaptiveThresholds = thresholds;
        }

        artifacts.HasConformalClassification = true;
        artifacts.ConformalClassificationThreshold = threshold;
        artifacts.ConformalClassificationNumClasses = numClasses;
    }

    private static Tensor<T> EnsureProbabilityTensor(Tensor<T> values, int batch, int classes)
    {
        if (classes <= 1 || batch <= 0)
        {
            return values;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var tolerance = 1e-3;
        var flat = values.ToVector();

        var looksLikeProbabilities = true;
        for (int b = 0; b < batch && looksLikeProbabilities; b++)
        {
            var sum = 0.0;
            var baseIndex = b * classes;
            for (int c = 0; c < classes; c++)
            {
                var p = numOps.ToDouble(flat[baseIndex + c]);
                if (p < 0.0 || p > 1.0)
                {
                    looksLikeProbabilities = false;
                    break;
                }
                sum += p;
            }

            if (Math.Abs(sum - 1.0) > tolerance)
            {
                looksLikeProbabilities = false;
            }
        }

        if (looksLikeProbabilities)
        {
            return values;
        }

        // Treat as logits and apply stable softmax per sample.
        var output = new Vector<T>(batch * classes);
        for (int b = 0; b < batch; b++)
        {
            var baseIndex = b * classes;
            var max = flat[baseIndex];
            for (int c = 1; c < classes; c++)
            {
                var v = flat[baseIndex + c];
                if (numOps.GreaterThan(v, max))
                {
                    max = v;
                }
            }

            var sumExp = numOps.Zero;
            for (int c = 0; c < classes; c++)
            {
                var ex = numOps.Exp(numOps.Subtract(flat[baseIndex + c], max));
                output[baseIndex + c] = ex;
                sumExp = numOps.Add(sumExp, ex);
            }

            for (int c = 0; c < classes; c++)
            {
                output[baseIndex + c] = numOps.Divide(output[baseIndex + c], sumExp);
            }
        }

        return new Tensor<T>([batch, classes], output).Reshape(values.Shape);
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

    private static ProbabilityCalibrationMethod ResolveCalibrationMethod(UncertaintyQuantificationOptions options, int numClasses)
    {
        if (options.CalibrationMethod != ProbabilityCalibrationMethod.Auto)
        {
            return options.CalibrationMethod;
        }

        if (numClasses == 2)
        {
            if (options.EnableIsotonicRegressionCalibration)
            {
                return ProbabilityCalibrationMethod.IsotonicRegression;
            }

            if (options.EnablePlattScaling)
            {
                return ProbabilityCalibrationMethod.PlattScaling;
            }
        }

        return options.EnableTemperatureScaling
            ? ProbabilityCalibrationMethod.TemperatureScaling
            : ProbabilityCalibrationMethod.None;
    }

    private static (T a, T b) FitPlattScalingBinary(Tensor<T> probabilities, Vector<int> labels, int batch)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var eps = 1e-12;

        double a = 1.0;
        double b = 0.0;

        var flat = probabilities.ToVector();

        for (int iter = 0; iter < 50; iter++)
        {
            double gradA = 0.0;
            double gradB = 0.0;
            double hAA = 0.0;
            double hAB = 0.0;
            double hBB = 0.0;

            for (int i = 0; i < batch; i++)
            {
                var p1 = numOps.ToDouble(flat[i * 2 + 1]);
                if (p1 < eps) p1 = eps;
                if (p1 > 1.0 - eps) p1 = 1.0 - eps;

                var logit = Math.Log(p1 / (1.0 - p1));
                var y = labels[i] == 1 ? 1.0 : 0.0;

                var z = a * logit + b;
                var p = 1.0 / (1.0 + Math.Exp(-z));

                var diff = p - y;
                gradA += diff * logit;
                gradB += diff;

                var w = p * (1.0 - p);
                hAA += w * logit * logit;
                hAB += w * logit;
                hBB += w;
            }

            // Regularize to avoid singular Hessian
            const double ridge = 1e-8;
            hAA += ridge;
            hBB += ridge;

            var det = hAA * hBB - hAB * hAB;
            if (Math.Abs(det) < 1e-18)
            {
                break;
            }

            var invAA = hBB / det;
            var invAB = -hAB / det;
            var invBB = hAA / det;

            var stepA = invAA * gradA + invAB * gradB;
            var stepB = invAB * gradA + invBB * gradB;

            a -= stepA;
            b -= stepB;

            if (Math.Abs(stepA) + Math.Abs(stepB) < 1e-8)
            {
                break;
            }
        }

        return (numOps.FromDouble(a), numOps.FromDouble(b));
    }

    private static Tensor<T> ApplyPlattScalingToProbabilitiesBinary(Tensor<T> probabilities, T a, T b, int batch)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var eps = 1e-12;
        var flat = probabilities.ToVector();
        var output = new Vector<T>(batch * 2);

        for (int i = 0; i < batch; i++)
        {
            var p1 = numOps.ToDouble(flat[i * 2 + 1]);
            if (p1 < eps) p1 = eps;
            if (p1 > 1.0 - eps) p1 = 1.0 - eps;

            var logit = Math.Log(p1 / (1.0 - p1));
            var z = numOps.Add(numOps.Multiply(a, numOps.FromDouble(logit)), b);
            var zDouble = numOps.ToDouble(z);
            var p1Cal = 1.0 / (1.0 + Math.Exp(-zDouble));

            var p1T = numOps.FromDouble(p1Cal);
            output[i * 2 + 1] = p1T;
            output[i * 2] = numOps.Subtract(numOps.One, p1T);
        }

        return new Tensor<T>([batch, 2], output).Reshape(probabilities.Shape);
    }

    private static (Vector<T> x, Vector<T> y) FitIsotonicCalibrationBinary(Tensor<T> probabilities, Vector<int> labels, int batch)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var eps = 1e-12;
        var flat = probabilities.ToVector();

        var pairs = new (double p, double y)[batch];
        for (int i = 0; i < batch; i++)
        {
            var p1 = numOps.ToDouble(flat[i * 2 + 1]);
            if (p1 < eps) p1 = eps;
            if (p1 > 1.0 - eps) p1 = 1.0 - eps;
            pairs[i] = (p1, labels[i] == 1 ? 1.0 : 0.0);
        }

        Array.Sort(pairs, (a, b) => a.p.CompareTo(b.p));

        var sums = new List<double>();
        var counts = new List<int>();
        var maxX = new List<double>();

        for (int i = 0; i < pairs.Length; i++)
        {
            sums.Add(pairs[i].y);
            counts.Add(1);
            maxX.Add(pairs[i].p);

            while (sums.Count >= 2)
            {
                int last = sums.Count - 1;
                int prev = last - 1;

                var avgPrev = sums[prev] / counts[prev];
                var avgLast = sums[last] / counts[last];
                if (avgPrev <= avgLast)
                {
                    break;
                }

                sums[prev] += sums[last];
                counts[prev] += counts[last];
                maxX[prev] = maxX[last];

                sums.RemoveAt(last);
                counts.RemoveAt(last);
                maxX.RemoveAt(last);
            }
        }

        var x = new Vector<T>(sums.Count);
        var y = new Vector<T>(sums.Count);
        for (int i = 0; i < sums.Count; i++)
        {
            x[i] = numOps.FromDouble(maxX[i]);
            y[i] = numOps.FromDouble(sums[i] / counts[i]);
        }

        return (x, y);
    }

    private static Tensor<T> ApplyIsotonicCalibrationToProbabilitiesBinary(Tensor<T> probabilities, Vector<T> x, Vector<T> y, int batch)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var eps = 1e-12;
        var flat = probabilities.ToVector();
        var output = new Vector<T>(batch * 2);

        for (int i = 0; i < batch; i++)
        {
            var p1 = numOps.ToDouble(flat[i * 2 + 1]);
            if (p1 < eps) p1 = eps;
            if (p1 > 1.0 - eps) p1 = 1.0 - eps;

            var p1T = numOps.FromDouble(p1);
            var p1Cal = EvaluateIsotonic(p1T, x, y, numOps);
            output[i * 2 + 1] = p1Cal;
            output[i * 2] = numOps.Subtract(numOps.One, p1Cal);
        }

        return new Tensor<T>([batch, 2], output).Reshape(probabilities.Shape);
    }

    private static T EvaluateIsotonic(T p, Vector<T> x, Vector<T> y, INumericOperations<T> numOps)
    {
        if (x.Length == 0)
        {
            return p;
        }

        var spanX = x.AsSpan();
        var spanY = y.AsSpan();
        for (int i = 0; i < spanX.Length; i++)
        {
            if (numOps.LessThanOrEquals(p, spanX[i]))
            {
                return spanY[i];
            }
        }

        return spanY[spanY.Length - 1];
    }

    private static T ComputeCrossConformalClassificationThreshold(Vector<T> scores, double confidenceLevel, int folds, int? seed)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var n = scores.Length;
        if (n == 0)
        {
            return numOps.Zero;
        }

        folds = Math.Max(2, folds);
        if (folds > n)
        {
            folds = n;
        }

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSeededRandom(42);
        var indices = Enumerable.Range(0, n).OrderBy(_ => rng.Next()).ToArray();

        var thresholds = new Vector<T>(folds);
        var thresholdCount = 0;

        for (int f = 0; f < folds; f++)
        {
            int start = (int)Math.Floor(f * (double)n / folds);
            int end = (int)Math.Floor((f + 1) * (double)n / folds);
            int len = end - start;
            if (len <= 0)
            {
                continue;
            }

            var foldScores = new Vector<T>(len);
            for (int i = 0; i < len; i++)
            {
                foldScores[i] = scores[indices[start + i]];
            }

            thresholds[thresholdCount++] = ComputeConformalClassificationThreshold(foldScores, confidenceLevel);
        }

        if (thresholdCount == 0)
        {
            return ComputeConformalClassificationThreshold(scores.Clone(), confidenceLevel);
        }

        var usable = thresholdCount == thresholds.Length ? thresholds : thresholds.Subvector(0, thresholdCount);
        var medianIndex = usable.Length / 2;
        return SelectKthInPlace(usable, medianIndex);
    }

    private static (double[] edges, Vector<T> thresholds) ComputeAdaptiveConformalClassificationThresholds(
        Vector<T> scores,
        Vector<T> confidence,
        double confidenceLevel,
        int bins)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var n = scores.Length;
        bins = Math.Max(1, bins);

        var edges = new double[bins + 1];
        for (int i = 0; i <= bins; i++)
        {
            edges[i] = i / (double)bins;
        }

        var globalThreshold = ComputeConformalClassificationThreshold(scores.Clone(), confidenceLevel);

        var buckets = new List<T>[bins];
        for (int i = 0; i < bins; i++)
        {
            buckets[i] = new List<T>();
        }

        for (int i = 0; i < n; i++)
        {
            var conf = numOps.ToDouble(confidence[i]);
            if (conf < 0.0) conf = 0.0;
            if (conf > 1.0) conf = 1.0;

            var bin = (int)Math.Floor(conf * bins);
            if (bin == bins) bin = bins - 1;

            buckets[bin].Add(scores[i]);
        }

        var thresholds = new Vector<T>(bins);
        for (int b = 0; b < bins; b++)
        {
            if (buckets[b].Count == 0)
            {
                thresholds[b] = globalThreshold;
                continue;
            }

            var bucketScores = new Vector<T>(buckets[b].Count);
            for (int i = 0; i < buckets[b].Count; i++)
            {
                bucketScores[i] = buckets[b][i];
            }

            thresholds[b] = ComputeConformalClassificationThreshold(bucketScores, confidenceLevel);
        }

        return (edges, thresholds);
    }

    private static void TryComputeLaplacePosteriorArtifacts(
        PredictionModelResult<T, TInput, TOutput> result,
        TInput xCalibration,
        Vector<int> labels,
        UncertaintyQuantificationOptions options,
        UncertaintyCalibrationArtifacts<T> artifacts)
    {
        if (!TryPreparePosteriorCalibrationDataForClassification(result, xCalibration, labels, options, out var input, out var target))
        {
            return;
        }

        var model = result.Model;
        if (model == null)
        {
            return;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var priorPrecision = numOps.FromDouble(options.LaplacePriorPrecision);

        Vector<T> gradients;
        try
        {
            gradients = model.ComputeGradients(input, target, lossFunction: null);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Warning: Laplace posterior fitting skipped because gradients could not be computed. Error: {ex.Message}");
            return;
        }

        var hDiag = new Vector<T>(gradients.Length);
        for (int i = 0; i < hDiag.Length; i++)
        {
            hDiag[i] = priorPrecision;
        }

        for (int i = 0; i < gradients.Length; i++)
        {
            var g2 = numOps.Multiply(gradients[i], gradients[i]);
            hDiag[i] = numOps.Add(hDiag[i], g2);
        }

        var variance = new Vector<T>(hDiag.Length);
        for (int i = 0; i < hDiag.Length; i++)
        {
            variance[i] = numOps.Divide(numOps.One, hDiag[i]);
        }

        artifacts.HasLaplacePosterior = true;
        artifacts.LaplacePosteriorMean = model.GetParameters().Clone();
        artifacts.LaplacePosteriorVarianceDiag = variance;
    }

    private static void TryComputeLaplacePosteriorArtifacts(
        PredictionModelResult<T, TInput, TOutput> result,
        TInput xCalibration,
        TOutput yCalibration,
        UncertaintyQuantificationOptions options,
        UncertaintyCalibrationArtifacts<T> artifacts)
    {
        if (!TryPreparePosteriorCalibrationDataForRegression(xCalibration, yCalibration, options, out var input, out var target))
        {
            return;
        }

        var model = result.Model;
        if (model == null)
        {
            return;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var priorPrecision = numOps.FromDouble(options.LaplacePriorPrecision);

        Vector<T> gradients;
        try
        {
            gradients = model.ComputeGradients(input, target, lossFunction: null);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Warning: Laplace posterior fitting skipped because gradients could not be computed. Error: {ex.Message}");
            return;
        }

        var hDiag = new Vector<T>(gradients.Length);
        for (int i = 0; i < hDiag.Length; i++)
        {
            hDiag[i] = priorPrecision;
        }

        for (int i = 0; i < gradients.Length; i++)
        {
            var g2 = numOps.Multiply(gradients[i], gradients[i]);
            hDiag[i] = numOps.Add(hDiag[i], g2);
        }

        var variance = new Vector<T>(hDiag.Length);
        for (int i = 0; i < hDiag.Length; i++)
        {
            variance[i] = numOps.Divide(numOps.One, hDiag[i]);
        }

        artifacts.HasLaplacePosterior = true;
        artifacts.LaplacePosteriorMean = model.GetParameters().Clone();
        artifacts.LaplacePosteriorVarianceDiag = variance;
    }

    private static void TryComputeSwagPosteriorArtifacts(
        PredictionModelResult<T, TInput, TOutput> result,
        TInput xCalibration,
        Vector<int> labels,
        UncertaintyQuantificationOptions options,
        UncertaintyCalibrationArtifacts<T> artifacts)
    {
        if (!TryPreparePosteriorCalibrationDataForClassification(result, xCalibration, labels, options, out var input, out var target))
        {
            return;
        }

        var model = result.Model;
        if (model == null)
        {
            return;
        }

        IFullModel<T, TInput, TOutput> swagModel;
        try
        {
            swagModel = model.Clone();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Warning: SWAG posterior fitting skipped because model cloning failed. Error: {ex.Message}");
            return;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var lr = numOps.FromDouble(options.SwagLearningRate);
        var steps = Math.Max(0, options.SwagNumSteps);
        var burnIn = Math.Max(0, options.SwagBurnInSteps);
        var snapshotsTarget = Math.Max(1, options.SwagNumSnapshots);
        var interval = Math.Max(1, (steps - burnIn) / snapshotsTarget);

        var mean = (Vector<T>?)null;
        var sqMean = (Vector<T>?)null;
        var snapshotCount = 0;

        for (int step = 0; step < steps; step++)
        {
            Vector<T> gradients;
            try
            {
                gradients = swagModel.ComputeGradients(input, target, lossFunction: null);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Warning: SWAG posterior fitting skipped because gradients could not be computed. Error: {ex.Message}");
                return;
            }

            swagModel.ApplyGradients(gradients, lr);

            if (step < burnIn)
            {
                continue;
            }

            if (((step - burnIn) % interval) != 0)
            {
                continue;
            }

            var parameters = swagModel.GetParameters();
            snapshotCount++;

            if (mean == null)
            {
                mean = parameters.Clone();
                sqMean = new Vector<T>(parameters.Length);
                for (int i = 0; i < parameters.Length; i++)
                {
                    sqMean[i] = numOps.Multiply(parameters[i], parameters[i]);
                }
                continue;
            }

            var k = numOps.FromDouble(snapshotCount);
            for (int i = 0; i < parameters.Length; i++)
            {
                mean[i] = numOps.Add(mean[i], numOps.Divide(numOps.Subtract(parameters[i], mean[i]), k));
                var p2 = numOps.Multiply(parameters[i], parameters[i]);
                sqMean![i] = numOps.Add(sqMean[i], numOps.Divide(numOps.Subtract(p2, sqMean[i]), k));
            }
        }

        if (mean == null || sqMean == null || snapshotCount == 0)
        {
            return;
        }

        var variance = new Vector<T>(mean.Length);
        for (int i = 0; i < mean.Length; i++)
        {
            var m2 = numOps.Multiply(mean[i], mean[i]);
            var v = numOps.Subtract(sqMean[i], m2);
            if (numOps.LessThan(v, numOps.Zero))
            {
                v = numOps.Zero;
            }
            variance[i] = v;
        }

        artifacts.HasSwagPosterior = true;
        artifacts.SwagPosteriorMean = mean;
        artifacts.SwagPosteriorVarianceDiag = variance;
    }

    private static void TryComputeSwagPosteriorArtifacts(
        PredictionModelResult<T, TInput, TOutput> result,
        TInput xCalibration,
        TOutput yCalibration,
        UncertaintyQuantificationOptions options,
        UncertaintyCalibrationArtifacts<T> artifacts)
    {
        if (!TryPreparePosteriorCalibrationDataForRegression(xCalibration, yCalibration, options, out var input, out var target))
        {
            return;
        }

        var model = result.Model;
        if (model == null)
        {
            return;
        }

        IFullModel<T, TInput, TOutput> swagModel;
        try
        {
            swagModel = model.Clone();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Warning: SWAG posterior fitting skipped because model cloning failed. Error: {ex.Message}");
            return;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var lr = numOps.FromDouble(options.SwagLearningRate);
        var steps = Math.Max(0, options.SwagNumSteps);
        var burnIn = Math.Max(0, options.SwagBurnInSteps);
        var snapshotsTarget = Math.Max(1, options.SwagNumSnapshots);
        var interval = Math.Max(1, (steps - burnIn) / snapshotsTarget);

        var mean = (Vector<T>?)null;
        var sqMean = (Vector<T>?)null;
        var snapshotCount = 0;

        for (int step = 0; step < steps; step++)
        {
            Vector<T> gradients;
            try
            {
                gradients = swagModel.ComputeGradients(input, target, lossFunction: null);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Warning: SWAG posterior fitting skipped because gradients could not be computed. Error: {ex.Message}");
                return;
            }

            swagModel.ApplyGradients(gradients, lr);

            if (step < burnIn)
            {
                continue;
            }

            if (((step - burnIn) % interval) != 0)
            {
                continue;
            }

            var parameters = swagModel.GetParameters();
            snapshotCount++;

            if (mean == null)
            {
                mean = parameters.Clone();
                sqMean = new Vector<T>(parameters.Length);
                for (int i = 0; i < parameters.Length; i++)
                {
                    sqMean[i] = numOps.Multiply(parameters[i], parameters[i]);
                }
                continue;
            }

            var k = numOps.FromDouble(snapshotCount);
            for (int i = 0; i < parameters.Length; i++)
            {
                mean[i] = numOps.Add(mean[i], numOps.Divide(numOps.Subtract(parameters[i], mean[i]), k));
                var p2 = numOps.Multiply(parameters[i], parameters[i]);
                sqMean![i] = numOps.Add(sqMean[i], numOps.Divide(numOps.Subtract(p2, sqMean[i]), k));
            }
        }

        if (mean == null || sqMean == null || snapshotCount == 0)
        {
            return;
        }

        var variance = new Vector<T>(mean.Length);
        for (int i = 0; i < mean.Length; i++)
        {
            var m2 = numOps.Multiply(mean[i], mean[i]);
            var v = numOps.Subtract(sqMean[i], m2);
            if (numOps.LessThan(v, numOps.Zero))
            {
                v = numOps.Zero;
            }
            variance[i] = v;
        }

        artifacts.HasSwagPosterior = true;
        artifacts.SwagPosteriorMean = mean;
        artifacts.SwagPosteriorVarianceDiag = variance;
    }

    private static bool TryPreparePosteriorCalibrationDataForClassification(
        PredictionModelResult<T, TInput, TOutput> result,
        TInput xCalibration,
        Vector<int> labels,
        UncertaintyQuantificationOptions options,
        out TInput input,
        out TOutput target)
    {
        input = default!;
        target = default!;

        if (labels.Length == 0 || result.Model == null)
        {
            return false;
        }

        var predicted = result.Predict(xCalibration);
        var probsTensor = ConversionsHelper.ConvertToTensor<T>(predicted!).Clone();
        var classes = probsTensor.Shape[probsTensor.Shape.Length - 1];
        if (classes <= 1)
        {
            return false;
        }

        var batch = probsTensor.Rank == 1 ? 1 : probsTensor.Length / classes;
        if (batch <= 0)
        {
            return false;
        }

        var take = Math.Min(batch, labels.Length);
        take = Math.Min(take, Math.Max(1, options.PosteriorFitMaxSamples));

        if (!TrySliceFirstSamples(xCalibration, take, out input))
        {
            System.Diagnostics.Debug.WriteLine($"Warning: Laplace/SWAG posterior fitting skipped because TInput '{typeof(TInput).Name}' could not be sliced.");
            return false;
        }

        if (!TryCreateOneHotTarget(labels, take, classes, out target))
        {
            System.Diagnostics.Debug.WriteLine($"Warning: Laplace/SWAG posterior fitting skipped because TOutput '{typeof(TOutput).Name}' one-hot targets are not supported for batch size {take}.");
            return false;
        }

        return true;
    }

    private static bool TryPreparePosteriorCalibrationDataForRegression(
        TInput xCalibration,
        TOutput yCalibration,
        UncertaintyQuantificationOptions options,
        out TInput input,
        out TOutput target)
    {
        input = default!;
        target = default!;

        var max = Math.Max(1, options.PosteriorFitMaxSamples);
        var take = max;

        if (xCalibration is Tensor<T> xTensor)
        {
            take = Math.Min(take, xTensor.Shape[0]);
        }
        else if (xCalibration is Matrix<T> xMatrix)
        {
            take = Math.Min(take, xMatrix.Rows);
        }
        else if (xCalibration is Vector<T>)
        {
            take = 1;
        }

        if (!TrySliceFirstSamples(xCalibration, take, out input))
        {
            System.Diagnostics.Debug.WriteLine($"Warning: Laplace/SWAG posterior fitting skipped because TInput '{typeof(TInput).Name}' could not be sliced.");
            return false;
        }

        if (!TrySliceFirstSamples(yCalibration, take, out target))
        {
            System.Diagnostics.Debug.WriteLine($"Warning: Laplace/SWAG posterior fitting skipped because TOutput '{typeof(TOutput).Name}' could not be sliced.");
            return false;
        }

        return true;
    }

    private static bool TrySliceFirstSamples<TValue>(TValue value, int take, out TValue sliced)
    {
        sliced = default!;

        if (take <= 0)
        {
            return false;
        }

        if (value is Tensor<T> tensor)
        {
            sliced = (TValue)(object)TakeFirstBatch(tensor, take);
            return true;
        }

        if (value is Matrix<T> matrix)
        {
            sliced = (TValue)(object)TakeFirstRows(matrix, take);
            return true;
        }

        if (value is Vector<T> vector)
        {
            var actual = Math.Min(take, vector.Length);
            if (actual == vector.Length)
            {
                sliced = value;
                return true;
            }

            var newVec = new Vector<T>(actual);
            for (int i = 0; i < actual; i++)
            {
                newVec[i] = vector[i];
            }

            sliced = (TValue)(object)newVec;
            return true;
        }

        if (take == 1)
        {
            sliced = value;
            return true;
        }

        return false;
    }

    private static bool TryCreateOneHotTarget(Vector<int> labels, int batch, int classes, out TOutput target)
    {
        target = default!;
        var numOps = MathHelper.GetNumericOperations<T>();

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            var flat = new Vector<T>(batch * classes);
            for (int i = 0; i < batch; i++)
            {
                var label = labels[i];
                if (label >= 0 && label < classes)
                {
                    flat[i * classes + label] = numOps.One;
                }
            }

            target = (TOutput)(object)new Tensor<T>([batch, classes], flat);
            return true;
        }

        if (typeof(TOutput) == typeof(Matrix<T>))
        {
            var mat = new Matrix<T>(batch, classes);
            for (int i = 0; i < batch; i++)
            {
                var label = labels[i];
                if (label >= 0 && label < classes)
                {
                    mat[i, label] = numOps.One;
                }
            }

            target = (TOutput)(object)mat;
            return true;
        }

        if (typeof(TOutput) == typeof(Vector<T>))
        {
            if (batch != 1)
            {
                return false;
            }

            var vec = new Vector<T>(classes);
            var label = labels[0];
            if (label >= 0 && label < classes)
            {
                vec[label] = numOps.One;
            }

            target = (TOutput)(object)vec;
            return true;
        }

        return false;
    }

    private static Matrix<T> TakeFirstRows(Matrix<T> matrix, int rows)
    {
        var take = Math.Min(rows, matrix.Rows);
        var result = new Matrix<T>(take, matrix.Columns);
        for (int r = 0; r < take; r++)
        {
            for (int c = 0; c < matrix.Columns; c++)
            {
                result[r, c] = matrix[r, c];
            }
        }

        return result;
    }

    private static Tensor<T> TakeFirstBatch(Tensor<T> tensor, int batch)
    {
        if (tensor.Rank == 1)
        {
            var take = Math.Min(batch, tensor.Shape[0]);
            var vec = tensor.ToVector();
            var sliced = new Vector<T>(take);
            for (int i = 0; i < take; i++)
            {
                sliced[i] = vec[i];
            }
            return new Tensor<T>([take], sliced);
        }

        var totalBatch = tensor.Shape[0];
        var takeBatch = Math.Min(batch, totalBatch);
        var rowSize = tensor.Length / totalBatch;

        var vecAll = tensor.ToVector();
        var slicedVec = new Vector<T>(takeBatch * rowSize);
        for (int i = 0; i < takeBatch * rowSize; i++)
        {
            slicedVec[i] = vecAll[i];
        }

        var newShape = new int[tensor.Shape.Length];
        newShape[0] = takeBatch;
        for (int d = 1; d < tensor.Shape.Length; d++)
        {
            newShape[d] = tensor.Shape[d];
        }

        return new Tensor<T>(newShape, slicedVec);
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
