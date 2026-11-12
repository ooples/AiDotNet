using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.AutoML.NAS;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet;

#pragma warning disable CS8601 // Possible null reference assignment in out params

public partial class AiModelBuilder<T, TInput, TOutput>
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
    /// var result = await new AiModelBuilder&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;()
    ///     .ConfigureDataLoader(loader)
    ///     .ConfigureModel(model)
    ///     .ConfigureOptimizer(optimizer)
    ///     .ConfigureUncertaintyQuantification()
    ///     .BuildAsync();
    /// </code>
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureUncertaintyQuantification(
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
    private void TryComputeAndAttachUncertaintyCalibrationArtifacts(AiModelResult<T, TInput, TOutput> result)
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
        else if (calibrationData is { HasTargets: true } && calibrationData.Y is TOutput yCalibration)
        {
            TryComputeRegressionConformalArtifacts(result, calibrationData.X, yCalibration, artifacts);

            if (effectiveMethod == UncertaintyQuantificationMethod.LaplaceApproximation)
            {
                TryComputeLaplacePosteriorArtifacts(result, calibrationData.X, yCalibration, _uncertaintyQuantificationOptions, artifacts);
            }
            else if (effectiveMethod == UncertaintyQuantificationMethod.Swag)
            {
                TryComputeSwagPosteriorArtifacts(result, calibrationData.X, yCalibration, _uncertaintyQuantificationOptions, artifacts);
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
        AiModelResult<T, TInput, TOutput> result,
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

        var engine = AiDotNetEngine.Current;
        var residuals = engine.Abs(engine.Subtract(actual, predicted));

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

        return KthSmallest(residuals, index);
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
        AiModelResult<T, TInput, TOutput> result,
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
        var probsTensor = ConversionsHelper.ConvertToTensor<T>((probsOutput ?? throw new InvalidOperationException("probsOutput is null where it is required."))).Clone();
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

        var engine = AiDotNetEngine.Current;
        var flatCalibrated = calibrated.ToVector();
        var calibratedRows = Tensor<T>.FromVector(flatCalibrated, new[] { batch, numClasses });

        // Predicted class = first maximum of each row; confidence = that maximum.
        var predictions = Vector<int>.FromArray(NasSamplingHelper.ArgMaxRows(calibratedRows, out T[] bestScores));
        var confidence = Vector<T>.FromArray(bestScores);

        var ece = new AiDotNet.UncertaintyQuantification.Calibration.ExpectedCalibrationError<T>(numBins: 10);
        artifacts.HasExpectedCalibrationError = true;
        artifacts.ExpectedCalibrationError = ece.Compute(confidence, predictions, labels);

        int[] labelValues = labels.ToArray();
        foreach (int invalidIndex in Enumerable.Range(0, batch).Where(i => labelValues[i] < 0 || labelValues[i] >= numClasses))
        {
            System.Diagnostics.Debug.WriteLine(
                $"Warning: Conformal classification skipped an invalid label at index {invalidIndex}: {labelValues[invalidIndex]}. Valid range is [0, {numClasses - 1}].");
        }

        int[] validRows = Enumerable.Range(0, batch).Where(i => labelValues[i] >= 0 && labelValues[i] < numClasses).ToArray();
        var validCount = validRows.Length;
        if (validCount == 0)
        {
            System.Diagnostics.Debug.WriteLine(
                "Warning: Conformal classification threshold could not be computed because no valid labels were provided.");
            return;
        }

        // True-class probability of every sample with a valid label, in sample order.
        var trueClassIndex = Tensor<int>.FromVector(Vector<int>.FromArray(validRows.Select(i => i * numClasses + labelValues[i]).ToArray()));
        var validRowIndex = Tensor<int>.FromVector(Vector<int>.FromArray(validRows));
        var validScores = engine.TensorGather(flatCalibrated.AsTensor(), trueClassIndex, 0).ToVector();
        var scoreConfidence = engine.TensorGather(confidence.AsTensor(), validRowIndex, 0).ToVector();

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
        var engine = AiDotNetEngine.Current;
        var tolerance = numOps.FromDouble(1e-3);
        var rows = Tensor<T>.FromVector(values.ToVector(), new[] { batch, classes });

        // Probabilities: every value in [0, 1] and every row summing to 1 within 1e-3. Comparisons are false for
        // NaN, so NaN entries do not disqualify (as in the scalar check).
        var outOfRange = engine.TensorAdd(engine.TensorLessThan(rows, numOps.Zero), engine.TensorGreaterThan(rows, numOps.One));
        var rowSums = engine.ReduceSum(rows, new[] { 1 }, false);
        var badSums = engine.TensorGreaterThan(engine.TensorAbs(engine.TensorSubtractScalar(rowSums, numOps.One)), tolerance);
        var looksLikeProbabilities = !numOps.GreaterThan(engine.TensorSum(outOfRange), numOps.Zero)
            && !numOps.GreaterThan(engine.TensorSum(badSums), numOps.Zero);

        if (looksLikeProbabilities)
        {
            return values;
        }

        // Treat as logits and apply stable softmax per sample.
        return engine.TensorSoftmax(rows, 1).Reshape(values._shape);
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
        var eps = ProbabilityFloor();
        var sumTolerance = numOps.FromDouble(1e-3);
        var uniformProbability = numOps.Divide(numOps.One, numOps.FromDouble(classes));
        var engine = AiDotNetEngine.Current;
        var shape = new[] { batch, classes };
        var rows = Tensor<T>.FromVector(probabilities.ToVector(), shape);
        var zeros = new Tensor<T>(shape);

        // Per row: negatives -> 0; a row summing below eps becomes uniform; a row off 1 by more than the tolerance
        // is renormalised; every probability is floored at eps before the log.
        var clipped = engine.TensorWhere(engine.TensorLessThan(rows, numOps.Zero), zeros, rows);
        var rowSums = engine.ReduceSum(clipped, new[] { 1 }, true);
        var fallbackToUniform = engine.TensorLessThan(rowSums, eps);
        var shouldNormalize = engine.TensorMultiply(
            engine.TensorAddScalar(engine.TensorNegate(fallbackToUniform), numOps.One),
            engine.TensorGreaterThan(engine.TensorAbs(engine.TensorSubtractScalar(rowSums, numOps.One)), sumTolerance));
        var normalized = engine.TensorWhere(engine.TensorBroadcastTo(shouldNormalize, shape),
            engine.TensorDivide(clipped, engine.TensorBroadcastTo(rowSums, shape)), clipped);
        var uniform = new Tensor<T>(shape);
        uniform.Fill(uniformProbability);
        var chosen = engine.TensorWhere(engine.TensorBroadcastTo(fallbackToUniform, shape), uniform, normalized);
        var floor = new Tensor<T>(shape);
        floor.Fill(eps);
        var logits = engine.TensorLog(engine.TensorWhere(engine.TensorLessThan(chosen, eps), floor, chosen)).ToMatrix();

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
        var engine = AiDotNetEngine.Current;
        var eps = ProbabilityFloor();
        var shape = new[] { batch, classes };
        var rows = Tensor<T>.FromVector(probabilities.ToVector(), shape);

        // softmax(log(max(p, eps)) / temperature) per row.
        var floor = new Tensor<T>(shape);
        floor.Fill(eps);
        var logits = engine.TensorDivideScalar(
            engine.TensorLog(engine.TensorWhere(engine.TensorLessThan(rows, eps), floor, rows)), temperature);
        return engine.TensorSoftmax(logits, 1).Reshape(probabilities._shape);
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

    internal static (T a, T b) FitPlattScalingBinary(Tensor<T> probabilities, Vector<int> labels, int batch)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;

        // Platt (1999) by Newton's method on the logistic log-loss of sigmoid(a * logit(p1) + b), with a 1e-8 ridge
        // on the Hessian diagonal; at most 50 steps, stopping when |step_a| + |step_b| < 1e-8 or the Hessian is
        // singular.
        var logit = PositiveClassLogit(probabilities, batch);
        var target = engine.TensorEquals(labels.AsTensor().Cast<T>(), numOps.One);
        var ones = new Tensor<T>(new[] { batch, 1 });
        ones.Fill(numOps.One);
        var design = Tensor<T>.Concatenate(new[] { logit.Reshape(new[] { batch, 1 }), ones }, 1);
        var designT = engine.TensorTranspose(design);
        var ridge = engine.TensorMultiplyScalar(engine.TensorEye<T>(2), numOps.FromDouble(1e-8));
        var singular = numOps.FromDouble(1e-18);
        var converged = numOps.FromDouble(1e-8);
        var none = numOps.Zero;
        var comparer = EqualityComparer<T>.Default;

        var theta = Tensor<T>.FromVector(new Vector<T>(new[] { numOps.One, numOps.Zero }), new[] { 2, 1 });
        for (int iter = 0; iter < 50; iter++)
        {
            var p = engine.TensorSigmoid(engine.TensorMatMul(design, theta).Reshape(new[] { batch }));
            var gradient = engine.TensorMatMul(designT, engine.TensorSubtract(p, target).Reshape(new[] { batch, 1 }));
            var w = engine.TensorMultiply(p, engine.TensorAddScalar(engine.TensorNegate(p), numOps.One));
            var weightedDesign = engine.TensorMultiply(design, engine.TensorBroadcastTo(w.Reshape(new[] { batch, 1 }), new[] { batch, 2 }));
            var hessian = engine.TensorAdd(engine.TensorMatMul(designT, weightedDesign), ridge);

            // Stop on a singular Hessian (|det| < 1e-18); the flag is a 0/1 engine comparison.
            var isSingular = engine.TensorSum(engine.TensorLessThan(engine.TensorAbs(LinalgDispatch<T>.Det(hessian)), singular));
            if (!comparer.Equals(isSingular, none))
            {
                break;
            }

            var step = LinalgDispatch<T>.Solve(hessian, gradient);
            theta = engine.TensorSubtract(theta, step);

            var stepNorm = Tensor<T>.FromScalar(engine.TensorSum(engine.TensorAbs(step)));
            if (!comparer.Equals(engine.TensorSum(engine.TensorLessThan(stepNorm, converged)), none))
            {
                break;
            }
        }

        var fitted = theta.ToVector();
        return (fitted[0], fitted[1]);
    }

    /// <summary>
    /// The smallest probability floor, starting at 1e-12, that is still positive in <typeparamref name="T"/>.
    /// 1e-12 underflows to 0 in Half, so a fixed 1e-12 floor would leave log(0) = -infinity there.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("Probes the representable margin of T on one scalar, a few iterations at most; no tensor data.")]
    internal static T ProbabilityFloor()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double eps = 1e-12;
        while (!numOps.GreaterThan(numOps.FromDouble(eps), numOps.Zero))
        {
            eps *= 2.0;
        }

        return numOps.FromDouble(eps);
    }

    /// <summary>
    /// The smallest clip margin, starting at 1e-12, for which both eps and 1 - eps are representable in
    /// <typeparamref name="T"/> as values strictly inside (0, 1). 1 - 1e-12 rounds to 1 in float (and 1e-12 to 0
    /// in Half), which would make logit(p) infinite; the margin grows to about the type's machine epsilon there.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("Probes the representable margin of T on one scalar, a few iterations at most; no tensor data.")]
    internal static (T Low, T High) ProbabilityClipBounds()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double eps = 1e-12;
        while (!(numOps.GreaterThan(numOps.FromDouble(eps), numOps.Zero)
            && numOps.LessThan(numOps.FromDouble(1.0 - eps), numOps.One)))
        {
            eps *= 2.0;
        }

        return (numOps.FromDouble(eps), numOps.FromDouble(1.0 - eps));
    }

    /// <summary>
    /// logit(p1) of the positive-class column of a [batch, 2] probability tensor, with p1 clipped to
    /// [eps, 1 - eps] (<see cref="ProbabilityClipBounds"/>), so the logit stays finite in every T.
    /// </summary>
    internal static Tensor<T> PositiveClassLogit(Tensor<T> probabilities, int batch)
    {
        var engine = AiDotNetEngine.Current;
        var (low, high) = ProbabilityClipBounds();
        var p1 = engine.TensorClamp(
            Tensor<T>.FromVector(probabilities.ToVector(), new[] { batch, 2 }).Slice(1, 1, 2).Reshape(new[] { batch }),
            low, high);
        return engine.TensorLog(engine.TensorDivide(p1, engine.TensorAddScalar(engine.TensorNegate(p1), numOps.One)));
    }

    private static Tensor<T> ApplyPlattScalingToProbabilitiesBinary(Tensor<T> probabilities, T a, T b, int batch)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;

        // p1' = sigmoid(a * logit(p1) + b), p0' = 1 - p1'.
        var p1 = engine.TensorSigmoid(engine.TensorAddScalar(engine.TensorMultiplyScalar(PositiveClassLogit(probabilities, batch), a), b));
        var p0 = engine.TensorAddScalar(engine.TensorNegate(p1), numOps.One);
        return Tensor<T>.Stack(new[] { p0, p1 }, 1).Reshape(probabilities._shape);
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
        var engine = AiDotNetEngine.Current;
        var (low, high) = ProbabilityClipBounds();
        var p1 = engine.TensorClamp(
            Tensor<T>.FromVector(probabilities.ToVector(), new[] { batch, 2 }).Slice(1, 1, 2).Reshape(new[] { batch }),
            low, high);

        // Step function of the isotonic fit: y at the first knot x >= p, the last y beyond the last knot, and the
        // identity when there is no fit.
        var p1Cal = p1;
        if (x.Length > 0)
        {
            var knot = engine.TensorSearchSorted(x.AsTensor(), p1, false);
            var levels = Tensor<T>.Concatenate(new[] { y.AsTensor(), y.GetSubVector(y.Length - 1, 1).AsTensor() }, 0);
            p1Cal = engine.TensorGather(levels, knot, 0);
        }

        var p0 = engine.TensorAddScalar(engine.TensorNegate(p1Cal), numOps.One);
        return Tensor<T>.Stack(new[] { p0, p1Cal }, 1).Reshape(probabilities._shape);
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

        var engine = AiDotNetEngine.Current;
        var thresholds = new List<T>(folds);

        for (int f = 0; f < folds; f++)
        {
            int start = (int)Math.Floor(f * (double)n / folds);
            int end = (int)Math.Floor((f + 1) * (double)n / folds);
            int len = end - start;
            if (len <= 0)
            {
                continue;
            }

            var foldIndex = Tensor<int>.FromVector(Vector<int>.FromArray(indices.Skip(start).Take(len).ToArray()));
            var foldScores = engine.TensorGather(scores.AsTensor(), foldIndex, 0).ToVector();
            thresholds.Add(ComputeConformalClassificationThreshold(foldScores, confidenceLevel));
        }

        if (thresholds.Count == 0)
        {
            return ComputeConformalClassificationThreshold(scores.Clone(), confidenceLevel);
        }

        var usable = Vector<T>.FromArray(thresholds.ToArray());
        var medianIndex = usable.Length / 2;
        return KthSmallest(usable, medianIndex);
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

        // bin = floor(clamp(confidence, 0, 1) * bins), with confidence 1 in the last bin.
        var engine = AiDotNetEngine.Current;
        int[] binOf = n == 0
            ? Array.Empty<int>()
            : engine.TensorMin(
                engine.TensorFloor(engine.TensorMultiplyScalar(
                    engine.TensorClamp(confidence.GetSubVector(0, n).AsTensor(), numOps.Zero, numOps.One), numOps.FromDouble(bins))),
                numOps.FromDouble(bins - 1)).Cast<int>().ToArray();

        var thresholds = new List<T>(bins);
        for (int b = 0; b < bins; b++)
        {
            int bin = b;
            int[] members = Enumerable.Range(0, n).Where(i => binOf[i] == bin).ToArray();
            if (members.Length == 0)
            {
                thresholds.Add(globalThreshold);
                continue;
            }

            var bucketScores = engine.TensorGather(scores.AsTensor(), Tensor<int>.FromVector(Vector<int>.FromArray(members)), 0).ToVector();
            thresholds.Add(ComputeConformalClassificationThreshold(bucketScores, confidenceLevel));
        }

        return (edges, Vector<T>.FromArray(thresholds.ToArray()));
    }

    private static void TryComputeLaplacePosteriorArtifacts(
        AiModelResult<T, TInput, TOutput> result,
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

        Vector<T> fisherDiagonal;
        try
        {
            fisherDiagonal = EmpiricalFisherDiagonal(InterfaceGuard.GradientComputable(model), input, target);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Warning: Laplace posterior fitting skipped because gradients could not be computed. Error: {ex.Message}");
            return;
        }

        // Diagonal Laplace (Daxberger et al. 2021): precision = prior precision + sum_i g_i^2, variance = 1 / precision.
        var engine = AiDotNetEngine.Current;
        var hDiag = engine.TensorAddScalar(fisherDiagonal.AsTensor(), priorPrecision);
        var variance = engine.Reciprocal(hDiag.ToVector());

        artifacts.HasLaplacePosterior = true;
        artifacts.LaplacePosteriorMean = InterfaceGuard.Parameterizable(model).GetParameters().Clone();
        artifacts.LaplacePosteriorVarianceDiag = variance;
    }

    private static void TryComputeLaplacePosteriorArtifacts(
        AiModelResult<T, TInput, TOutput> result,
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

        Vector<T> fisherDiagonal;
        try
        {
            fisherDiagonal = EmpiricalFisherDiagonal(InterfaceGuard.GradientComputable(model), input, target);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Warning: Laplace posterior fitting skipped because gradients could not be computed. Error: {ex.Message}");
            return;
        }

        // Diagonal Laplace (Daxberger et al. 2021): precision = prior precision + sum_i g_i^2, variance = 1 / precision.
        var engine = AiDotNetEngine.Current;
        var hDiag = engine.TensorAddScalar(fisherDiagonal.AsTensor(), priorPrecision);
        var variance = engine.Reciprocal(hDiag.ToVector());

        artifacts.HasLaplacePosterior = true;
        artifacts.LaplacePosteriorMean = InterfaceGuard.Parameterizable(model).GetParameters().Clone();
        artifacts.LaplacePosteriorVarianceDiag = variance;
    }

    private static void TryComputeSwagPosteriorArtifacts(
        AiModelResult<T, TInput, TOutput> result,
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
                gradients = InterfaceGuard.GradientComputable(swagModel).ComputeGradients(input, target, lossFunction: null);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Warning: SWAG posterior fitting skipped because gradients could not be computed. Error: {ex.Message}");
                return;
            }

            InterfaceGuard.GradientComputable(swagModel).ApplyGradients(gradients, lr);

            if (step < burnIn)
            {
                continue;
            }

            if (((step - burnIn) % interval) != 0)
            {
                continue;
            }

            var parameters = InterfaceGuard.Parameterizable(swagModel).GetParameters();
            snapshotCount++;

            var engine = AiDotNetEngine.Current;
            var theta = parameters.AsTensor();
            var thetaSquared = engine.TensorMultiply(theta, theta);
            if (mean == null)
            {
                mean = parameters.Clone();
                sqMean = thetaSquared.ToVector();
                continue;
            }

            // Running first and second moments (SWAG-Diagonal, Maddox et al. 2019): m += (x - m) / k.
            var k = numOps.FromDouble(snapshotCount);
            var currentSqMean = sqMean ?? throw new InvalidOperationException("sqMean not initialized.");
            mean = engine.TensorAdd(mean.AsTensor(), engine.TensorDivideScalar(engine.TensorSubtract(theta, mean.AsTensor()), k)).ToVector();
            sqMean = engine.TensorAdd(currentSqMean.AsTensor(),
                engine.TensorDivideScalar(engine.TensorSubtract(thetaSquared, currentSqMean.AsTensor()), k)).ToVector();
        }

        if (mean == null || sqMean == null || snapshotCount == 0)
        {
            return;
        }

        // Diagonal variance max(E[x^2] - E[x]^2, 0); NaN stays NaN as in the scalar loop.
        var varianceEngine = AiDotNetEngine.Current;
        var meanTensor = mean.AsTensor();
        var rawVariance = varianceEngine.TensorSubtract(sqMean.AsTensor(), varianceEngine.TensorMultiply(meanTensor, meanTensor));
        var variance = varianceEngine.TensorWhere(varianceEngine.TensorLessThan(rawVariance, numOps.Zero),
            new Tensor<T>(new[] { rawVariance.Length }), rawVariance).ToVector();

        artifacts.HasSwagPosterior = true;
        artifacts.SwagPosteriorMean = mean;
        artifacts.SwagPosteriorVarianceDiag = variance;
    }

    private static void TryComputeSwagPosteriorArtifacts(
        AiModelResult<T, TInput, TOutput> result,
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
                gradients = InterfaceGuard.GradientComputable(swagModel).ComputeGradients(input, target, lossFunction: null);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Warning: SWAG posterior fitting skipped because gradients could not be computed. Error: {ex.Message}");
                return;
            }

            InterfaceGuard.GradientComputable(swagModel).ApplyGradients(gradients, lr);

            if (step < burnIn)
            {
                continue;
            }

            if (((step - burnIn) % interval) != 0)
            {
                continue;
            }

            var parameters = InterfaceGuard.Parameterizable(swagModel).GetParameters();
            snapshotCount++;

            var engine = AiDotNetEngine.Current;
            var theta = parameters.AsTensor();
            var thetaSquared = engine.TensorMultiply(theta, theta);
            if (mean == null)
            {
                mean = parameters.Clone();
                sqMean = thetaSquared.ToVector();
                continue;
            }

            // Running first and second moments (SWAG-Diagonal, Maddox et al. 2019): m += (x - m) / k.
            var k = numOps.FromDouble(snapshotCount);
            var currentSqMean = sqMean ?? throw new InvalidOperationException("sqMean not initialized.");
            mean = engine.TensorAdd(mean.AsTensor(), engine.TensorDivideScalar(engine.TensorSubtract(theta, mean.AsTensor()), k)).ToVector();
            sqMean = engine.TensorAdd(currentSqMean.AsTensor(),
                engine.TensorDivideScalar(engine.TensorSubtract(thetaSquared, currentSqMean.AsTensor()), k)).ToVector();
        }

        if (mean == null || sqMean == null || snapshotCount == 0)
        {
            return;
        }

        // Diagonal variance max(E[x^2] - E[x]^2, 0); NaN stays NaN as in the scalar loop.
        var varianceEngine = AiDotNetEngine.Current;
        var meanTensor = mean.AsTensor();
        var rawVariance = varianceEngine.TensorSubtract(sqMean.AsTensor(), varianceEngine.TensorMultiply(meanTensor, meanTensor));
        var variance = varianceEngine.TensorWhere(varianceEngine.TensorLessThan(rawVariance, numOps.Zero),
            new Tensor<T>(new[] { rawVariance.Length }), rawVariance).ToVector();

        artifacts.HasSwagPosterior = true;
        artifacts.SwagPosteriorMean = mean;
        artifacts.SwagPosteriorVarianceDiag = variance;
    }

    private static bool TryPreparePosteriorCalibrationDataForClassification(
        AiModelResult<T, TInput, TOutput> result,
        TInput xCalibration,
        Vector<int> labels,
        UncertaintyQuantificationOptions options,
        out TInput input,
        out TOutput target)
    {
        input = default;
        target = default;

        if (labels.Length == 0 || result.Model == null)
        {
            return false;
        }

        var predicted = result.Predict(xCalibration);
        var probsTensor = ConversionsHelper.ConvertToTensor<T>((predicted ?? throw new InvalidOperationException("predicted is null where it is required."))).Clone();
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
        input = default;
        target = default;

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
        sliced = default;

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

            sliced = (TValue)(object)vector.GetSubVector(0, actual);
            return true;
        }

        if (take == 1)
        {
            sliced = value;
            return true;
        }

        return false;
    }

    /// <summary>
    /// Diagonal of the empirical Fisher, sum over samples of the squared per-sample loss gradient g_i^2 (the
    /// diagonal "EF" Hessian approximation of laplace-torch; Daxberger et al. 2021, Sec. 2). The squared gradient
    /// of the whole batch is not this: it drops every per-sample term that cancels in the sum.
    /// </summary>
    internal static Vector<T> EmpiricalFisherDiagonal(IGradientComputable<T, TInput, TOutput> model, TInput input, TOutput target)
    {
        var engine = AiDotNetEngine.Current;
        int samples = LeadingSampleCount(input);
        Tensor<T>? fisher = null;
        foreach (int i in Enumerable.Range(0, samples))
        {
            var sampleInput = samples == 1 ? input : SliceSample(input, i);
            var sampleTarget = samples == 1 ? target : SliceSample(target, i);
            var g = model.ComputeGradients(sampleInput, sampleTarget, lossFunction: null).AsTensor();
            var squared = engine.TensorMultiply(g, g);
            fisher = fisher is null ? squared : engine.TensorAdd(fisher, squared);
        }

        return fisher is null ? new Vector<T>(0) : fisher.ToVector();
    }

    private static int LeadingSampleCount<TValue>(TValue value)
    {
        return value switch
        {
            Tensor<T> tensor when tensor.Rank >= 2 => tensor.Shape[0],
            Matrix<T> matrix => matrix.Rows,
            _ => 1
        };
    }

    /// <summary>
    /// Sample <paramref name="index"/> of a batched value, keeping its rank: [1, ...] for a tensor, a one-row
    /// matrix, or a one-element vector (one target per sample).
    /// </summary>
    private static TValue SliceSample<TValue>(TValue value, int index)
    {
        object sample = value switch
        {
            // A rank-1 [N] target (one scalar per sample) slices to [1], like a Vector target.
            Tensor<T> tensor when tensor.Rank >= 1 => tensor.Slice(0, index, index + 1).Contiguous(),
            Matrix<T> matrix => Matrix<T>.FromRowVectors(new[] { matrix.GetRow(index) }),
            Vector<T> vector => vector.GetSubVector(index, 1),
            _ => throw new NotSupportedException($"Per-sample gradients need a Tensor, Matrix or Vector batch, not {typeof(TValue).Name}.")
        };

        return (TValue)sample;
    }

    private static bool TryCreateOneHotTarget(Vector<int> labels, int batch, int classes, out TOutput target)
    {
        target = default;
        var numOps = MathHelper.GetNumericOperations<T>();

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            target = (TOutput)(object)OneHotRows(labels, batch, classes);
            return true;
        }

        if (typeof(TOutput) == typeof(Matrix<T>))
        {
            target = (TOutput)(object)OneHotRows(labels, batch, classes).ToMatrix();
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

    /// <summary>
    /// A [batch, classes] one-hot tensor of the first <paramref name="batch"/> labels; rows whose label is outside
    /// [0, classes) stay all zero.
    /// </summary>
    private static Tensor<T> OneHotRows(Vector<int> labels, int batch, int classes)
    {
        var engine = AiDotNetEngine.Current;
        int[] values = labels.ToArray().Take(batch).ToArray();
        var safeLabels = Tensor<int>.FromVector(Vector<int>.FromArray(values.Select(l => l >= 0 && l < classes ? l : 0).ToArray()));
        var validRows = Tensor<int>.FromVector(Vector<int>.FromArray(values.Select(l => l >= 0 && l < classes ? 1 : 0).ToArray()))
            .Cast<T>().Reshape(new[] { batch, 1 });
        var oneHot = engine.TensorOneHot<T>(safeLabels, classes).Reshape(new[] { batch, classes });
        return engine.TensorMultiply(oneHot, engine.TensorBroadcastTo(validRows, new[] { batch, classes }));
    }

    private static Matrix<T> TakeFirstRows(Matrix<T> matrix, int rows)
    {
        var take = Math.Min(rows, matrix.Rows);
        var result = new Matrix<T>(take, matrix.Columns);
        matrix.AsSpan().Slice(0, take * matrix.Columns).CopyTo(result.AsWritableSpan());
        return result;
    }

    private static Tensor<T> TakeFirstBatch(Tensor<T> tensor, int batch)
    {
        if (tensor.Rank == 1)
        {
            var take = Math.Min(batch, tensor.Shape[0]);
            return new Tensor<T>([take], tensor.ToVector().GetSubVector(0, take));
        }

        var totalBatch = tensor.Shape[0];
        var takeBatch = Math.Min(batch, totalBatch);
        var rowSize = tensor.Length / totalBatch;

        var slicedVec = tensor.ToVector().GetSubVector(0, takeBatch * rowSize);

        var newShape = new int[tensor.Shape.Length];
        newShape[0] = takeBatch;
        for (int d = 1; d < tensor.Shape.Length; d++)
        {
            newShape[d] = tensor.Shape[d];
        }

        return new Tensor<T>(newShape, slicedVec);
    }

    /// <summary>
    /// Split-conformal threshold on the true-class probability (LAC; Sadinle et al. 2019): with nonconformity
    /// 1 - p, the conformal quantile is the ceil((n + 1)(1 - alpha))-th smallest nonconformity, i.e. the
    /// floor((n + 1) alpha)-th smallest true-class probability. The set {c : p_c >= threshold} then covers the
    /// true class with probability at least 1 - alpha. When floor((n + 1) alpha) is 0 every class is kept.
    /// </summary>
    /// <param name="scores">Scores for the true labels (order does not matter).</param>
    /// <param name="confidenceLevel">Desired coverage probability (e.g., 0.9).</param>
    /// <returns>The score threshold.</returns>
    internal static T ComputeConformalClassificationThreshold(Vector<T> scores, double confidenceLevel)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var n = scores.Length;
        if (n == 0)
        {
            return numOps.Zero;
        }

        var alpha = Math.Min(1.0, Math.Max(0.0, 1.0 - confidenceLevel));

        // The small slack keeps (n + 1) * alpha from landing just below an integer through rounding of 1 - level.
        var rank = (int)Math.Floor((n + 1) * alpha + 1e-9);
        if (rank < 1)
        {
            return numOps.Zero;
        }

        return KthSmallest(scores, Math.Min(rank, n) - 1);
    }

    /// <summary>
    /// The k-th smallest value (0-based) of <paramref name="values"/>; the vector is left unchanged.
    /// </summary>
    private static T KthSmallest(Vector<T> values, int k)
    {
        var (sorted, _) = AiDotNetEngine.Current.TensorSort(values.AsTensor(), 0, false);
        return sorted.ToVector()[k];
    }

}
