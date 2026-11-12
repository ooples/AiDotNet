using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Calibration;

/// <summary>
/// Calibrates model outputs to produce reliable probability estimates.
/// </summary>
/// <remarks>
/// <para>
/// Probability calibration transforms raw model scores into well-calibrated probabilities.
/// A well-calibrated model means: if you see 1000 predictions of "70% probability", about
/// 700 of those should be positive outcomes.
/// </para>
/// <para>
/// <b>For Beginners:</b> Calibration is like adjusting a thermometer. Your model might
/// always say "90 degrees" when the true temperature is 85. Calibration learns this
/// systematic error and corrects for it.
///
/// <b>Available methods:</b>
/// - Platt Scaling: Fits a logistic curve - good for SVMs and many classifiers
/// - Isotonic Regression: Non-parametric curve fitting - more flexible but needs more data
/// - Temperature Scaling: Simple division - popular for neural networks
/// - Histogram Binning: Averages within bins - simple and interpretable
/// - Beta Calibration: Fits beta distribution - good for bounded outputs
///
/// <b>Example usage:</b>
/// 1. Train your model and get predicted probabilities
/// 2. Fit the calibrator on a held-out validation set
/// 3. Use calibrator to transform all future predictions
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ProbabilityCalibrator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Platt scaling parameters (A and B in sigmoid(Ax + B)).
    /// </summary>
    private T _plattA;
    private T _plattB;

    /// <summary>
    /// Isotonic regression points.
    /// </summary>
    private List<(T x, T y)>? _isotonicPoints;

    /// <summary>
    /// Temperature parameter for temperature scaling.
    /// </summary>
    private T _temperature;

    /// <summary>
    /// Beta calibration parameters.
    /// </summary>
    private T _betaA;
    private T _betaB;
    private T _betaC;

    /// <summary>
    /// Histogram bin edges and probabilities.
    /// </summary>
    private T[]? _binEdges;
    private T[]? _binProbabilities;

    /// <summary>
    /// Stored calibration data for Venn-ABERS per-point dual isotonic fits.
    /// </summary>
    private T[]? _vennCalibrationScores;
    private T[]? _vennCalibrationLabels;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly ProbabilityCalibratorOptions _options;

    /// <summary>
    /// Whether the calibrator has been fitted.
    /// </summary>
    private bool _isFitted;

    /// <summary>
    /// Initializes a new instance of the calibrator.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public ProbabilityCalibrator(ProbabilityCalibratorOptions? options = null)
    {
        _options = options ?? new ProbabilityCalibratorOptions();
        _plattA = NumOps.FromDouble(-1);
        _plattB = NumOps.Zero;
        _temperature = NumOps.One;
        _betaA = NumOps.One;
        _betaB = NumOps.One;
        _betaC = NumOps.Zero;
        _isFitted = false;
    }

    /// <summary>
    /// Fits the calibrator to predicted scores and true labels.
    /// </summary>
    /// <param name="scores">Predicted scores/probabilities from the model.</param>
    /// <param name="labels">True binary labels (0 or 1).</param>
    public void Fit(Vector<T> scores, Vector<T> labels)
    {
        if (scores.Length != labels.Length)
        {
            throw new ArgumentException("Scores and labels must have the same length.");
        }

        var scoresArray = scores.ToArray();
        var labelsArray = labels.ToArray();

        switch (_options.CalibratorMethod)
        {
            case ProbabilityCalibrationMethod.PlattScaling:
                FitPlattScaling(scoresArray, labelsArray);
                break;

            case ProbabilityCalibrationMethod.IsotonicRegression:
                FitIsotonicRegression(scoresArray, labelsArray);
                break;

            case ProbabilityCalibrationMethod.TemperatureScaling:
                FitTemperatureScaling(scoresArray, labelsArray);
                break;

            case ProbabilityCalibrationMethod.BetaCalibration:
                FitBetaCalibration(scoresArray, labelsArray);
                break;

            case ProbabilityCalibrationMethod.HistogramBinning:
                FitHistogramBinning(scoresArray, labelsArray);
                break;

            case ProbabilityCalibrationMethod.BayesianBinning:
                FitBayesianBinning(scoresArray, labelsArray);
                break;

            case ProbabilityCalibrationMethod.VennABERS:
                FitVennABERS(scoresArray, labelsArray);
                break;

            case ProbabilityCalibrationMethod.None:
                // No calibration - identity transform
                break;

            default:
                throw new ArgumentOutOfRangeException(
                    nameof(_options.CalibratorMethod),
                    _options.CalibratorMethod,
                    "Unknown calibration method.");
        }

        _isFitted = true;
    }

    /// <summary>
    /// Transforms predicted scores into calibrated probabilities.
    /// </summary>
    /// <param name="scores">Predicted scores/probabilities from the model.</param>
    /// <returns>Calibrated probabilities.</returns>
    public Vector<T> Transform(Vector<T> scores)
    {
        if (!_isFitted)
        {
            throw new InvalidOperationException("Calibrator must be fitted before transform.");
        }

        var scoresArray = scores.ToArray();
        T[] calibrated;

        switch (_options.CalibratorMethod)
        {
            case ProbabilityCalibrationMethod.PlattScaling:
                calibrated = TransformPlattScaling(scoresArray);
                break;

            case ProbabilityCalibrationMethod.IsotonicRegression:
                calibrated = TransformIsotonicRegression(scoresArray);
                break;

            case ProbabilityCalibrationMethod.TemperatureScaling:
                calibrated = TransformTemperatureScaling(scoresArray);
                break;

            case ProbabilityCalibrationMethod.BetaCalibration:
                calibrated = TransformBetaCalibration(scoresArray);
                break;

            case ProbabilityCalibrationMethod.HistogramBinning:
            case ProbabilityCalibrationMethod.BayesianBinning:
                calibrated = TransformHistogramBinning(scoresArray);
                break;

            case ProbabilityCalibrationMethod.VennABERS:
                calibrated = TransformVennABERS(scoresArray);
                break;

            case ProbabilityCalibrationMethod.None:
                // Identity transform - return scores unchanged
                calibrated = scoresArray;
                break;

            default:
                throw new ArgumentOutOfRangeException(
                    nameof(_options.CalibratorMethod),
                    _options.CalibratorMethod,
                    "Unknown calibration method.");
        }

        return new Vector<T>(calibrated);
    }

    /// <summary>
    /// Fits the calibrator and transforms in one step.
    /// </summary>
    public Vector<T> FitTransform(Vector<T> scores, Vector<T> labels)
    {
        Fit(scores, labels);
        return Transform(scores);
    }

    /// <summary>
    /// Computes the Expected Calibration Error (ECE).
    /// </summary>
    /// <param name="scores">Predicted probabilities.</param>
    /// <param name="labels">True binary labels.</param>
    /// <param name="numBins">Number of bins for ECE calculation.</param>
    /// <returns>ECE value (lower is better, 0 is perfect calibration).</returns>
    public double ComputeECE(Vector<T> scores, Vector<T> labels, int numBins = 10)
    {
        int n = scores.Length;
        if (numBins <= 0) return 0;

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var (counts, avgProbs, avgAccs) = BinAverages(Tensor<T>.FromVector(scores), Tensor<T>.FromVector(labels), numBins);

        // Sum over non-empty bins of (count / n) * |avgProb - avgAcc|; empty bins have zero count
        var binWeights = engine.TensorDivideScalar(counts, NumOps.FromDouble(n));
        var gaps = engine.TensorAbs(engine.TensorSubtract(avgProbs, avgAccs));
        var nonEmpty = engine.TensorGreaterThan(counts, NumOps.Zero);
        var weighted = engine.TensorWhere(nonEmpty, engine.TensorMultiply(binWeights, gaps), engine.TensorZerosLike(gaps));
        T ece = engine.TensorSum(weighted);

        return NumOps.ToDouble(ece);
    }

    /// <summary>
    /// Computes the Maximum Calibration Error (MCE).
    /// </summary>
    public double ComputeMCE(Vector<T> scores, Vector<T> labels, int numBins = 10)
    {
        if (numBins <= 0) return 0;

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var (counts, avgProbs, avgAccs) = BinAverages(Tensor<T>.FromVector(scores), Tensor<T>.FromVector(labels), numBins);

        // Largest |avgProb - avgAcc| over non-empty bins, starting from 0; a NaN error never replaces the maximum
        var gaps = engine.TensorAbs(engine.TensorSubtract(avgProbs, avgAccs));
        var counted = EvaluationTensorOps<T>.And(
            engine.TensorGreaterThan(counts, NumOps.Zero), EvaluationTensorOps<T>.Not(EvaluationTensorOps<T>.NanMask(gaps)));
        T mce = engine.TensorMaxValue(engine.TensorWhere(counted, gaps, engine.TensorZerosLike(gaps)));

        return NumOps.ToDouble(mce);
    }

    /// <summary>
    /// Computes the Brier Score.
    /// </summary>
    /// <param name="scores">Predicted probabilities.</param>
    /// <param name="labels">True binary labels.</param>
    /// <returns>Brier score (lower is better, 0 is perfect).</returns>
    public double ComputeBrierScore(Vector<T> scores, Vector<T> labels)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var predicted = Tensor<T>.FromVector(scores);
        var actual = engine.TensorNarrow(Tensor<T>.FromVector(labels), 0, 0, scores.Length);
        T brier = engine.TensorSumOfSquares(engine.TensorSubtract(predicted, actual));

        return NumOps.ToDouble(NumOps.Divide(brier, NumOps.FromDouble(scores.Length)));
    }

    /// <summary>
    /// Gets a reliability diagram (calibration curve data).
    /// </summary>
    /// <param name="scores">Predicted probabilities.</param>
    /// <param name="labels">True binary labels.</param>
    /// <param name="numBins">Number of bins.</param>
    /// <returns>Tuple of (meanPredicted, fractionPositives, binCounts).</returns>
    public (T[] meanPredicted, T[] fractionPositives, int[] binCounts) GetReliabilityDiagram(
        Vector<T> scores, Vector<T> labels, int numBins = 10)
    {
        if (numBins <= 0)
            return (new T[numBins], new T[numBins], new int[numBins]);

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var (counts, avgPreds, avgLabels) = BinAverages(Tensor<T>.FromVector(scores), Tensor<T>.FromVector(labels), numBins);

        // Empty bins report the bin midpoint (lower + upper) / 2 and a zero fraction of positives
        var binCount = NumOps.FromDouble(numBins);
        var binIndex = engine.TensorArange<T>(0, numBins);
        var lower = engine.TensorDivideScalar(binIndex, binCount);
        var upper = engine.TensorDivideScalar(engine.TensorAddScalar(binIndex, NumOps.One), binCount);
        var midpoints = engine.TensorDivideScalar(engine.TensorAdd(lower, upper), NumOps.FromDouble(2));
        var nonEmpty = engine.TensorGreaterThan(counts, NumOps.Zero);

        var meanPredicted = engine.TensorWhere(nonEmpty, avgPreds, midpoints).ToArray();
        var fractionPositives = engine.TensorWhere(nonEmpty, avgLabels, engine.TensorZerosLike(avgLabels)).ToArray();
        var binCounts = counts.Cast<int>().ToArray();

        return (meanPredicted, fractionPositives, binCounts);
    }

    // Per-bin sample counts and mean score and mean label over the uniform bins [b/numBins, (b+1)/numBins) (the last
    // bin also holds 1.0). Empty bins get a zero count and zero means.
    private static (Tensor<T> Counts, Tensor<T> MeanScores, Tensor<T> MeanLabels) BinAverages(
        Tensor<T> scoreTensor, Tensor<T> labels, int numBins)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        int n = scoreTensor.Length;
        if (n == 0)
        {
            var zeros = new Tensor<T>(new[] { numBins });
            return (zeros, zeros, zeros);
        }

        var shape = new[] { numBins, n };
        var row = new[] { 1, n };
        var labelTensor = engine.TensorNarrow(labels, 0, 0, n);
        var inBin = EvaluationTensorOps<T>.UniformBinMask(scoreTensor, numBins);
        var zero = engine.TensorZerosLike(inBin);
        var binnedScores = engine.TensorWhere(inBin, engine.TensorBroadcastTo(engine.Reshape(scoreTensor, row), shape), zero);
        var binnedLabels = engine.TensorWhere(inBin, engine.TensorBroadcastTo(engine.Reshape(labelTensor, row), shape), zero);

        var counts = engine.ReduceSum(inBin, new[] { 1 }, false);
        var safeCounts = engine.TensorMax(counts, NumOps.One);
        return (counts,
            engine.TensorDivide(engine.ReduceSum(binnedScores, new[] { 1 }, false), safeCounts),
            engine.TensorDivide(engine.ReduceSum(binnedLabels, new[] { 1 }, false), safeCounts));
    }

    #region Platt Scaling

    private void FitPlattScaling(T[] scores, T[] labels)
    {
        int n = scores.Length;
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var scoreTensor = new Tensor<T>(scores, new[] { n });
        var isPositive = engine.TensorGreaterThan(new Tensor<T>(labels, new[] { labels.Length }), NumOps.FromDouble(0.5));
        int numPositive = EvaluationTensorOps<T>.Count(isPositive);
        int numNegative = n - numPositive;

        // Target values with smoothing
        double hiTarget = (numPositive + 1.0) / (numPositive + 2.0);
        double loTarget = 1.0 / (numNegative + 2.0);

        var targets = engine.TensorAdd(
            engine.TensorMultiplyScalar(isPositive, NumOps.FromDouble(hiTarget)),
            engine.TensorMultiplyScalar(EvaluationTensorOps<T>.Not(isPositive), NumOps.FromDouble(loTarget)));

        // Initialize A and B
        double A = 0;
        double B = Math.Log((numNegative + 1.0) / (numPositive + 1.0));

        // Newton-Raphson optimization: the sums run over the samples in T, the 2x2 step on the scalar sums
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            var (g1, g2, h11, h22, h21) = PlattNewtonSums(scoreTensor, targets, A, B);

            // Regularization
            h11 += _options.Regularization;
            h22 += _options.Regularization;

            // Check gradient
            if (Math.Abs(g1) < _options.Tolerance && Math.Abs(g2) < _options.Tolerance)
            {
                break;
            }

            // Newton update
            double det = h11 * h22 - h21 * h21;
            if (Math.Abs(det) < 1e-10) det = Math.Sign(det) == 0 ? 1e-10 : Math.Sign(det) * 1e-10;

            double dA = (g1 * h22 - g2 * h21) / det;
            double dB = (g2 * h11 - g1 * h21) / det;

            A += dA;
            B += dB;
        }

        _plattA = NumOps.FromDouble(A);
        _plattB = NumOps.FromDouble(B);
    }

    // Gradient and Hessian sums of the Platt log-likelihood at (A, B): with f = s A + B, p = 1 / (1 + e^f) and
    // q = 1 / (1 + e^-f) (both computed stably by the sigmoid), d1 = t - p and d2 = p q.
    private static (double G1, double G2, double H11, double H22, double H21) PlattNewtonSums(
        Tensor<T> scores, Tensor<T> targets, double a, double b)
    {
        var engine = AiDotNetEngine.Current;
        var fApB = engine.TensorAddScalar(engine.TensorMultiplyScalar(scores, NumOps.FromDouble(a)), NumOps.FromDouble(b));
        var p = engine.TensorSigmoid(engine.TensorNegate(fApB));
        var q = engine.TensorSigmoid(fApB);
        var d2 = engine.TensorMultiply(p, q);
        var d1 = engine.TensorSubtract(targets, p);
        var scaledD2 = engine.TensorMultiply(scores, d2);

        return (NumOps.ToDouble(engine.TensorSum(engine.TensorMultiply(scores, d1))),
            NumOps.ToDouble(engine.TensorSum(d1)),
            NumOps.ToDouble(engine.TensorSum(engine.TensorMultiply(scores, scaledD2))),
            NumOps.ToDouble(engine.TensorSum(d2)),
            NumOps.ToDouble(engine.TensorSum(scaledD2)));
    }

    private T[] TransformPlattScaling(T[] scores)
    {
        // p = 1 / (1 + exp(-(s A + B))), the stable sigmoid on both signs of s A + B
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var fApB = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(new Tensor<T>(scores, new[] { scores.Length }), _plattA), _plattB);
        return engine.TensorSigmoid(fApB).ToArray();
    }

    #endregion

    #region Isotonic Regression

    private void FitIsotonicRegression(T[] scores, T[] labels)
    {
        // Pool Adjacent Violators Algorithm (PAVA) - uses double internally for numerical stability
        var sorted = scores.Zip(labels, (s, l) => (score: s, label: NumOps.ToDouble(l)))
            .OrderBy(x => NumOps.ToDouble(x.score))
            .ToList();

        int n = sorted.Count;
        var y = sorted.Select(x => x.label).ToArray();
        var weights = Enumerable.Repeat(1.0, n).ToArray();

        // PAVA
        for (int i = 0; i < n - 1; i++)
        {
            if (y[i] > y[i + 1])
            {
                // Pool adjacent violators
                double pooledY = (y[i] * weights[i] + y[i + 1] * weights[i + 1]) / (weights[i] + weights[i + 1]);
                double pooledWeight = weights[i] + weights[i + 1];

                y[i] = pooledY;
                weights[i] = pooledWeight;

                // Remove i+1
                y = [.. y.Take(i + 1), .. y.Skip(i + 2)];
                weights = [.. weights.Take(i + 1), .. weights.Skip(i + 2)];
                n--;

                // Go back to check previous
                i = Math.Max(-1, i - 2);
            }
        }

        // Create calibration points (stored as T)
        _isotonicPoints = [];
        int idx = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double sumWeight = weights[i];
            while (sumWeight > 0)
            {
                _isotonicPoints.Add((sorted[idx].score, NumOps.FromDouble(y[i])));
                idx++;
                sumWeight--;
            }
        }
    }

    private T[] TransformIsotonicRegression(T[] scores)
    {
        if (_isotonicPoints == null || _isotonicPoints.Count == 0)
        {
            return scores;
        }

        int m = _isotonicPoints.Count;
        var pointX = new T[m];
        var pointY = new T[m];
        for (int i = 0; i < m; i++)
        {
            pointX[i] = _isotonicPoints[i].x;
            pointY[i] = _isotonicPoints[i].y;
        }

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        int count = scores.Length;
        var shape = new[] { count };
        var s = new Tensor<T>(scores, shape);
        var x = new Tensor<T>(pointX, new[] { m });
        var y = new Tensor<T>(pointY, new[] { m });
        var firstY = engine.TensorBroadcastTo(engine.TensorNarrow(y, 0, 0, 1), shape);

        // A single point maps every score (NaN included) to its value
        if (m == 1) return firstY.ToArray();

        // Bracketing points: hi is the first point with x > s (the binary search's upper end), clamped to
        // [1, m - 1]; a NaN score keeps the search's initial bracket (0, 1).
        var nanScore = EvaluationTensorOps<T>.NanMask(s);
        var nanIndex = nanScore.Cast<int>();
        var hi = engine.TensorClamp(engine.TensorSearchSorted(x, s, true), 1, m - 1);
        hi = engine.TensorAdd(engine.TensorSubtract(hi, engine.TensorMultiply(nanIndex, hi)), nanIndex);
        var lo = engine.TensorSubtractScalar(hi, 1);

        // Linear interpolation (or the lower value when the bracket has zero width)
        var x1 = engine.TensorGather(x, lo, 0);
        var x2 = engine.TensorGather(x, hi, 0);
        var y1 = engine.TensorGather(y, lo, 0);
        var y2 = engine.TensorGather(y, hi, 0);
        var diff = engine.TensorSubtract(x2, x1);
        var flat = engine.TensorLessThan(engine.TensorAbs(diff), NumOps.FromDouble(1e-10));
        var interpolated = engine.TensorAdd(y1, engine.TensorDivide(
            engine.TensorMultiply(engine.TensorSubtract(s, x1), engine.TensorSubtract(y2, y1)), diff));
        var inner = engine.TensorWhere(flat, y1, interpolated);

        // Scores at or below the first point take its value, then scores at or above the last point take the last value
        var finite = EvaluationTensorOps<T>.Not(nanScore);
        var firstX = engine.TensorBroadcastTo(engine.TensorNarrow(x, 0, 0, 1), shape);
        var lastX = engine.TensorBroadcastTo(engine.TensorNarrow(x, 0, m - 1, 1), shape);
        var lastY = engine.TensorBroadcastTo(engine.TensorNarrow(y, 0, m - 1, 1), shape);
        var atOrBelowFirst = EvaluationTensorOps<T>.And(finite, EvaluationTensorOps<T>.Not(engine.TensorGreaterThan(s, firstX)));
        var atOrAboveLast = EvaluationTensorOps<T>.And(finite, EvaluationTensorOps<T>.Not(engine.TensorLessThan(s, lastX)));

        return engine.TensorWhere(atOrBelowFirst, firstY, engine.TensorWhere(atOrAboveLast, lastY, inner)).ToArray();
    }

    #endregion

    #region Temperature Scaling

    private void FitTemperatureScaling(T[] scores, T[] labels)
    {
        // Temperature scaling for logits: p = sigmoid(logit / T)
        // If scores are already probabilities, convert to logits first
        // Internal computation in double for numerical stability (exp, log)

        using var noGrad = new NoGradScope<T>();
        var logits = TemperatureLogits(scores);
        var labelTensor = new Tensor<T>(labels, new[] { labels.Length });

        // Optimize temperature to minimize cross-entropy
        double temp = 1.0;
        double lr = _options.LearningRate;

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // Gradient of cross-entropy w.r.t. T:
            // dL/dT = -(p - y) * logit / T², averaged over the samples
            double gradient = TemperatureGradient(logits, labelTensor, temp);

            if (Math.Abs(gradient) < _options.Tolerance)
            {
                break;
            }

            temp -= lr * gradient;
            temp = Math.Max(0.01, temp);  // Keep T positive
        }

        _temperature = NumOps.FromDouble(temp);
    }

    // Scores clamped to [1e-10, 1 - 1e-10] (NaN stays NaN) and mapped to logits.
    private static Tensor<T> TemperatureLogits(T[] scores)
    {
        var clamped = EvaluationTensorOps<T>.ClampPreservingNaN(
            new Tensor<T>(scores, new[] { scores.Length }), NumOps.FromDouble(1e-10), NumOps.FromDouble(1 - 1e-10));
        return EvaluationTensorOps<T>.Logit(clamped);
    }

    private static double TemperatureGradient(Tensor<T> logits, Tensor<T> labels, double temp)
    {
        var engine = AiDotNetEngine.Current;
        var p = engine.TensorSigmoid(engine.TensorDivideScalar(logits, NumOps.FromDouble(temp)));
        double sum = NumOps.ToDouble(engine.TensorSum(engine.TensorMultiply(engine.TensorSubtract(labels, p), logits)));
        return sum / (temp * temp) / logits.Length;
    }

    private T[] TransformTemperatureScaling(T[] scores)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        return engine.TensorSigmoid(engine.TensorDivideScalar(TemperatureLogits(scores), _temperature)).ToArray();
    }

    #endregion

    #region Beta Calibration

    private void FitBetaCalibration(T[] scores, T[] labels)
    {
        // Beta calibration: p = 1 / (1 + 1/exp(a*log(s/(1-s)) + b*log(s) + c))
        // Simplified: use logistic regression on transformed features
        // Internal computation in double for numerical stability (exp, log)

        int n = scores.Length;
        using var noGrad = new NoGradScope<T>();
        var (logit, logScore) = BetaFeatures(scores);
        var labelTensor = new Tensor<T>(labels, new[] { labels.Length });

        // Logistic regression on (logit, log(s), 1)
        double[] w = [0, 0, 0];
        double lr = _options.LearningRate;

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            double[] grad = BetaGradient(logit, logScore, labelTensor, w);

            double maxGrad = grad.Max(Math.Abs);
            if (maxGrad < _options.Tolerance) break;

            for (int j = 0; j < 3; j++)
            {
                w[j] -= lr * grad[j] / n;
            }
        }

        _betaA = NumOps.FromDouble(w[0]);
        _betaB = NumOps.FromDouble(w[1]);
        _betaC = NumOps.FromDouble(w[2]);
    }

    // Scores clamped to [1e-6, 1 - 1e-6] (NaN stays NaN): the logit and log(s) features.
    private static (Tensor<T> Logit, Tensor<T> LogScore) BetaFeatures(T[] scores)
    {
        var clamped = EvaluationTensorOps<T>.ClampPreservingNaN(
            new Tensor<T>(scores, new[] { scores.Length }), NumOps.FromDouble(1e-6), NumOps.FromDouble(1 - 1e-6));
        return (EvaluationTensorOps<T>.Logit(clamped), AiDotNetEngine.Current.TensorLog(clamped));
    }

    // Sum over samples of (sigmoid(w0 logit + w1 log s + w2) - y) times each feature (the third feature is 1).
    private static double[] BetaGradient(Tensor<T> logit, Tensor<T> logScore, Tensor<T> labels, double[] w)
    {
        var engine = AiDotNetEngine.Current;
        var z = engine.TensorAddScalar(
            engine.TensorAdd(
                engine.TensorMultiplyScalar(logit, NumOps.FromDouble(w[0])),
                engine.TensorMultiplyScalar(logScore, NumOps.FromDouble(w[1]))),
            NumOps.FromDouble(w[2]));
        var error = engine.TensorSubtract(engine.TensorSigmoid(z), labels);
        return
        [
            NumOps.ToDouble(engine.TensorSum(engine.TensorMultiply(error, logit))),
            NumOps.ToDouble(engine.TensorSum(engine.TensorMultiply(error, logScore))),
            NumOps.ToDouble(engine.TensorSum(error))
        ];
    }

    private T[] TransformBetaCalibration(T[] scores)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var (logit, logScore) = BetaFeatures(scores);
        var z = engine.TensorAddScalar(
            engine.TensorAdd(engine.TensorMultiplyScalar(logit, _betaA), engine.TensorMultiplyScalar(logScore, _betaB)), _betaC);
        return engine.TensorSigmoid(z).ToArray();
    }

    #endregion

    #region Histogram Binning

    private void FitHistogramBinning(T[] scores, T[] labels)
    {
        int numBins = _options.NumBins;
        _binEdges = new T[numBins + 1];
        _binProbabilities = new T[numBins];

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var edges = engine.TensorDivideScalar(engine.TensorArange<T>(0, numBins + 1), NumOps.FromDouble(numBins));
        _binEdges = edges.ToArray();
        if (numBins == 0) return;

        // Mean label per bin [b/numBins, (b+1)/numBins) (the last bin also holds 1.0); empty bins take the midpoint
        var (counts, _, meanLabels) = BinAverages(
            new Tensor<T>(scores, new[] { scores.Length }), new Tensor<T>(labels, new[] { labels.Length }), numBins);
        var midpoints = engine.TensorDivideScalar(
            engine.TensorAdd(engine.TensorNarrow(edges, 0, 0, numBins), engine.TensorNarrow(edges, 0, 1, numBins)),
            NumOps.FromDouble(2));
        _binProbabilities = engine.TensorWhere(engine.TensorGreaterThan(counts, NumOps.Zero), meanLabels, midpoints).ToArray();
    }

    private void FitBayesianBinning(T[] scores, T[] labels)
    {
        // Bayesian Binning into Quantiles (BBQ):
        // Evaluate candidate binning schemes with varying number of bins,
        // score each by BIC (Bayesian Information Criterion),
        // and select the scheme with the highest score.
        int n = scores.Length;
        int maxBins = Math.Min(_options.NumBins, n);

        double bestBic = double.NegativeInfinity;
        T[]? bestEdges = null;
        T[]? bestProbs = null;

        if (maxBins >= 2)
        {
            var engine = AiDotNetEngine.Current;
            using var noGrad = new NoGradScope<T>();

            // Sort once (NaN first, then ascending, as OrderBy on doubles) and co-sort labels; prefix sums of the
            // sorted labels give each bin's positive count
            var scoreTensor = new Tensor<T>(scores, new[] { n });
            bool hasNaN = EvaluationTensorOps<T>.Count(EvaluationTensorOps<T>.NanMask(scoreTensor)) > 0;
            var sortedScores = EvaluationTensorOps<T>.SortNaNFirst(scoreTensor);
            var order = engine.TensorSort(scoreTensor).Indices;
            var sortedLabels = engine.TensorGather(new Tensor<T>(labels, new[] { labels.Length }), order, 0);
            var labelPrefix = engine.TensorConcatenate(
                new[] { new Tensor<T>(new[] { 1 }), engine.TensorCumSum(sortedLabels, 0) }, 0);

            for (int numBins = 2; numBins <= maxBins; numBins++)
            {
                var (bic, edges, probs) = ScoreQuantileBinning(sortedScores, labelPrefix, numBins, hasNaN);

                if (bic > bestBic)
                {
                    bestBic = bic;
                    bestEdges = edges;
                    bestProbs = probs;
                }
            }
        }

        if (bestEdges is not null && bestProbs is not null)
        {
            _binEdges = bestEdges;
            _binProbabilities = bestProbs;
        }
        else
        {
            // Fallback to simple histogram binning
            FitHistogramBinning(scores, labels);
        }
    }

    // One BBQ candidate: equal-frequency edges [0, s[b * (n / numBins)] for b = 1..numBins-1, 1] over the sorted
    // scores, bins filled in sorted order (score < upper edge, the last bin score <= 1), bin probability
    // positives / count clamped to [1e-10, 1 - 1e-10] (empty bins: the edge midpoint), and
    // BIC = binomial log-likelihood - numBins / 2 * ln n. A NaN score sorts first and stops the partition, so every
    // bin is then empty.
    private static (double Bic, T[] Edges, T[] Probs) ScoreQuantileBinning(
        Tensor<T> sortedScores, Tensor<T> labelPrefix, int numBins, bool hasNaN)
    {
        var engine = AiDotNetEngine.Current;
        int n = sortedScores.Length;
        int binSize = n / numBins;

        var innerIndices = new int[numBins - 1];
        for (int b = 1; b < numBins; b++)
            innerIndices[b - 1] = Math.Min(b * binSize, n - 1);

        var edges = engine.TensorConcatenate(new[]
        {
            new Tensor<T>(new[] { NumOps.Zero }, new[] { 1 }),
            engine.TensorGather(sortedScores, new Tensor<int>(innerIndices, new[] { numBins - 1 }), 0),
            new Tensor<T>(new[] { NumOps.One }, new[] { 1 })
        }, 0);
        var lowerEdges = engine.TensorNarrow(edges, 0, 0, numBins);
        var upperEdges = engine.TensorNarrow(edges, 0, 1, numBins);

        Tensor<int> ends;
        if (hasNaN)
        {
            ends = new Tensor<int>(new[] { numBins });
        }
        else
        {
            // Bin b ends at the first sorted score >= its upper edge (the last bin at the first score > 1); the
            // partition never moves backwards
            var belowUpper = engine.TensorSearchSorted(sortedScores, engine.TensorNarrow(upperEdges, 0, 0, numBins - 1), false);
            var atMostOne = engine.TensorSearchSorted(sortedScores, engine.TensorNarrow(upperEdges, 0, numBins - 1, 1), true);
            ends = engine.TensorCumMax(engine.TensorConcatenate(new[] { belowUpper, atMostOne }, 0), 0);
        }

        var starts = engine.TensorConcatenate(new[] { new Tensor<int>(new[] { 1 }), engine.TensorNarrow(ends, 0, 0, numBins - 1) }, 0);
        var counts = engine.TensorSubtract(ends, starts).Cast<T>();
        var positives = engine.TensorSubtract(engine.TensorGather(labelPrefix, ends, 0), engine.TensorGather(labelPrefix, starts, 0));
        var negatives = engine.TensorSubtract(counts, positives);
        var nonEmpty = engine.TensorGreaterThan(counts, NumOps.Zero);

        var p = EvaluationTensorOps<T>.ClampPreservingNaN(
            engine.TensorDivide(positives, engine.TensorMax(counts, NumOps.One)), NumOps.FromDouble(1e-10), NumOps.FromDouble(1 - 1e-10));
        var midpoints = engine.TensorDivideScalar(engine.TensorAdd(lowerEdges, upperEdges), NumOps.FromDouble(2.0));
        var probs = engine.TensorWhere(nonEmpty, p, midpoints);

        // Binomial log-likelihood of the non-empty bins
        var logLikelihoods = engine.TensorAdd(
            engine.TensorMultiply(positives, engine.TensorLog(p)),
            engine.TensorMultiply(negatives, engine.TensorLog(engine.TensorSubtract(engine.TensorOnesLike(p), p))));
        double logLikelihood = NumOps.ToDouble(engine.TensorSum(
            engine.TensorWhere(nonEmpty, logLikelihoods, engine.TensorZerosLike(logLikelihoods))));

        // BIC = log-likelihood - (k/2) * log(n), where k = numBins (one parameter per bin)
        double bic = logLikelihood - (numBins / 2.0) * Math.Log(n);
        return (bic, edges.ToArray(), probs.ToArray());
    }

    private T[] TransformHistogramBinning(T[] scores)
    {
        if (_binEdges == null || _binProbabilities == null)
        {
            return scores;
        }

        int numBins = _binProbabilities.Length;
        int count = scores.Length;
        if (count == 0) return [];
        if (numBins == 0)
            throw new IndexOutOfRangeException("Index was outside the bounds of the array.");

        // Each score takes the probability of the first bin with edges[b] <= s < edges[b + 1], else the last bin's
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var shape = new[] { numBins, count };
        var column = new[] { numBins, 1 };
        var scoreTensor = new Tensor<T>(scores, new[] { count });
        var s = engine.TensorBroadcastTo(engine.Reshape(scoreTensor, new[] { 1, count }), shape);
        var edges = new Tensor<T>(_binEdges, new[] { _binEdges.Length });
        var lower = engine.TensorBroadcastTo(engine.Reshape(engine.TensorNarrow(edges, 0, 0, numBins), column), shape);
        var upper = engine.TensorBroadcastTo(engine.Reshape(engine.TensorNarrow(edges, 0, 1, numBins), column), shape);

        var comparable = EvaluationTensorOps<T>.Not(EvaluationTensorOps<T>.Or(
            EvaluationTensorOps<T>.NanMask(s), EvaluationTensorOps<T>.NanMask(lower)));
        var atLeastLower = EvaluationTensorOps<T>.And(comparable, EvaluationTensorOps<T>.Not(engine.TensorLessThan(s, lower)));
        var matches = EvaluationTensorOps<T>.And(atLeastLower, engine.TensorLessThan(s, upper));
        var firstMatch = EvaluationTensorOps<T>.And(
            matches, EvaluationTensorOps<T>.EqualMask(engine.TensorCumSum(matches, 0), NumOps.One));

        var probabilities = new Tensor<T>(_binProbabilities, new[] { numBins });
        var binProbability = engine.TensorBroadcastTo(engine.Reshape(probabilities, column), shape);
        var picked = engine.ReduceSum(engine.TensorWhere(firstMatch, binProbability, engine.TensorZerosLike(binProbability)), new[] { 0 }, false);
        var anyMatch = engine.TensorGreaterThan(engine.ReduceSum(matches, new[] { 0 }, false), NumOps.Zero);
        var lastProbability = engine.TensorBroadcastTo(engine.TensorNarrow(probabilities, 0, numBins - 1, 1), new[] { count });

        return engine.TensorWhere(anyMatch, picked, lastProbability).ToArray();
    }

    #endregion

    #region Venn-ABERS

    private void FitVennABERS(T[] scores, T[] labels)
    {
        // Venn-ABERS produces calibrated probability intervals by fitting
        // isotonic regression twice for each test point (once assuming y=0, once y=1).
        // Here we pre-fit isotonic regression on the calibration set; the per-point
        // dual isotonic fits are applied at transform time.
        FitIsotonicRegression(scores, labels);
        // Store calibration data for per-point dual fits
        _vennCalibrationScores = scores;
        _vennCalibrationLabels = labels;
    }

    private T[] TransformVennABERS(T[] scores)
    {
        if (_vennCalibrationScores == null || _vennCalibrationLabels == null)
        {
            return TransformIsotonicRegression(scores);
        }

        // For each test point, fit two isotonic regressions using local state
        // to avoid mutating the shared _isotonicPoints field.
        // p0: assuming test point label = 0, p1: assuming test point label = 1
        // Final calibrated probability = p1 / (1 - p0 + p1)
        var savedIsotonicPoints = _isotonicPoints;
        var lowerFits = new T[scores.Length];
        var upperFits = new T[scores.Length];
        T zero = NumOps.Zero;
        T one = NumOps.One;

        try
        {
            for (int i = 0; i < scores.Length; i++)
            {
                // Augment calibration set with test point labeled 0
                var scores0 = _vennCalibrationScores.Append(scores[i]).ToArray();
                var labels0 = _vennCalibrationLabels.Append(zero).ToArray();
                FitIsotonicRegression(scores0, labels0);
                lowerFits[i] = TransformIsotonicRegression([scores[i]])[0];

                // Augment calibration set with test point labeled 1
                var scores1 = _vennCalibrationScores.Append(scores[i]).ToArray();
                var labels1 = _vennCalibrationLabels.Append(one).ToArray();
                FitIsotonicRegression(scores1, labels1);
                upperFits[i] = TransformIsotonicRegression([scores[i]])[0];
            }
        }
        finally
        {
            // Always restore the original isotonic state
            _isotonicPoints = savedIsotonicPoints;
        }

        // Combine to get point estimate: p1 / (1 - p0 + p1), or 0.5 when the denominator is not above 1e-10
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var shape = new[] { scores.Length };
        var p0 = new Tensor<T>(lowerFits, shape);
        var p1 = new Tensor<T>(upperFits, shape);
        var denom = engine.TensorAdd(engine.TensorSubtract(engine.TensorOnesLike(p0), p0), p1);
        var half = engine.TensorAddScalar(engine.TensorZerosLike(p0), NumOps.FromDouble(0.5));
        var calibrated = engine.TensorWhere(
            engine.TensorGreaterThan(denom, NumOps.FromDouble(1e-10)), engine.TensorDivide(p1, denom), half).ToArray();

        return calibrated;
    }

    #endregion

    /// <summary>
    /// Gets the calibration method being used.
    /// </summary>
    public ProbabilityCalibrationMethod ProbabilityCalibrationMethod => _options.CalibratorMethod;

    /// <summary>
    /// Gets the Platt scaling parameters (A, B).
    /// </summary>
    public (T a, T b) GetPlattParameters() => (_plattA, _plattB);

    /// <summary>
    /// Gets the temperature parameter.
    /// </summary>
    public T GetTemperature() => _temperature;

    /// <summary>
    /// Gets the beta calibration parameters (a, b, c).
    /// </summary>
    public (T a, T b, T c) GetBetaParameters() => (_betaA, _betaB, _betaC);
}
