using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

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
        T ece = NumOps.Zero;

        for (int b = 0; b < numBins; b++)
        {
            T lower = NumOps.FromDouble((double)b / numBins);
            T upper = NumOps.FromDouble((double)(b + 1) / numBins);

            // Include scores == 1.0 in the last bin
            var binIndices = Enumerable.Range(0, n)
                .Where(i => NumOps.GreaterThanOrEquals(scores[i], lower) && (b == numBins - 1 ? NumOps.LessThanOrEquals(scores[i], upper) : NumOps.LessThan(scores[i], upper)))
                .ToList();

            if (binIndices.Count > 0)
            {
                T sumProb = NumOps.Zero;
                T sumAcc = NumOps.Zero;
                foreach (var i in binIndices)
                {
                    sumProb = NumOps.Add(sumProb, scores[i]);
                    sumAcc = NumOps.Add(sumAcc, labels[i]);
                }
                T count = NumOps.FromDouble(binIndices.Count);
                T avgProb = NumOps.Divide(sumProb, count);
                T avgAcc = NumOps.Divide(sumAcc, count);
                T binWeight = NumOps.Divide(count, NumOps.FromDouble(n));

                ece = NumOps.Add(ece, NumOps.Multiply(binWeight, NumOps.Abs(NumOps.Subtract(avgProb, avgAcc))));
            }
        }

        return NumOps.ToDouble(ece);
    }

    /// <summary>
    /// Computes the Maximum Calibration Error (MCE).
    /// </summary>
    public double ComputeMCE(Vector<T> scores, Vector<T> labels, int numBins = 10)
    {
        int n = scores.Length;
        T mce = NumOps.Zero;

        for (int b = 0; b < numBins; b++)
        {
            T lower = NumOps.FromDouble((double)b / numBins);
            T upper = NumOps.FromDouble((double)(b + 1) / numBins);

            // Include scores == 1.0 in the last bin
            var binIndices = Enumerable.Range(0, n)
                .Where(i => NumOps.GreaterThanOrEquals(scores[i], lower) && (b == numBins - 1 ? NumOps.LessThanOrEquals(scores[i], upper) : NumOps.LessThan(scores[i], upper)))
                .ToList();

            if (binIndices.Count > 0)
            {
                T sumProb = NumOps.Zero;
                T sumAcc = NumOps.Zero;
                foreach (var i in binIndices)
                {
                    sumProb = NumOps.Add(sumProb, scores[i]);
                    sumAcc = NumOps.Add(sumAcc, labels[i]);
                }
                T count = NumOps.FromDouble(binIndices.Count);
                T avgProb = NumOps.Divide(sumProb, count);
                T avgAcc = NumOps.Divide(sumAcc, count);
                T absError = NumOps.Abs(NumOps.Subtract(avgProb, avgAcc));

                if (NumOps.GreaterThan(absError, mce))
                {
                    mce = absError;
                }
            }
        }

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
        T brier = NumOps.Zero;
        for (int i = 0; i < scores.Length; i++)
        {
            T diff = NumOps.Subtract(scores[i], labels[i]);
            brier = NumOps.Add(brier, NumOps.Multiply(diff, diff));
        }

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
        int n = scores.Length;
        var meanPredicted = new T[numBins];
        var fractionPositives = new T[numBins];
        var binCounts = new int[numBins];

        for (int b = 0; b < numBins; b++)
        {
            T lower = NumOps.FromDouble((double)b / numBins);
            T upper = NumOps.FromDouble((double)(b + 1) / numBins);

            // Include scores == 1.0 in the last bin
            var binIndices = Enumerable.Range(0, n)
                .Where(i => NumOps.GreaterThanOrEquals(scores[i], lower) && (b == numBins - 1 ? NumOps.LessThanOrEquals(scores[i], upper) : NumOps.LessThan(scores[i], upper)))
                .ToList();

            binCounts[b] = binIndices.Count;

            if (binIndices.Count > 0)
            {
                T sumPred = NumOps.Zero;
                T sumLabel = NumOps.Zero;
                foreach (var i in binIndices)
                {
                    sumPred = NumOps.Add(sumPred, scores[i]);
                    sumLabel = NumOps.Add(sumLabel, labels[i]);
                }
                T count = NumOps.FromDouble(binIndices.Count);
                meanPredicted[b] = NumOps.Divide(sumPred, count);
                fractionPositives[b] = NumOps.Divide(sumLabel, count);
            }
            else
            {
                meanPredicted[b] = NumOps.Divide(NumOps.Add(lower, upper), NumOps.FromDouble(2));
                fractionPositives[b] = NumOps.Zero;
            }
        }

        return (meanPredicted, fractionPositives, binCounts);
    }

    #region Platt Scaling

    private void FitPlattScaling(T[] scores, T[] labels)
    {
        int n = scores.Length;
        int numPositive = labels.Count(l => NumOps.ToDouble(l) > 0.5);
        int numNegative = n - numPositive;

        // Target values with smoothing
        double hiTarget = (numPositive + 1.0) / (numPositive + 2.0);
        double loTarget = 1.0 / (numNegative + 2.0);

        var targets = labels.Select(l => NumOps.ToDouble(l) > 0.5 ? hiTarget : loTarget).ToArray();

        // Initialize A and B
        double A = 0;
        double B = Math.Log((numNegative + 1.0) / (numPositive + 1.0));

        // Newton-Raphson optimization (internal computation in double for numerical stability)
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            double g1 = 0, g2 = 0, h11 = 0, h22 = 0, h21 = 0;

            for (int i = 0; i < n; i++)
            {
                double si = NumOps.ToDouble(scores[i]);
                double fApB = si * A + B;
                double p, q;

                if (fApB >= 0)
                {
                    p = Math.Exp(-fApB) / (1 + Math.Exp(-fApB));
                    q = 1 / (1 + Math.Exp(-fApB));
                }
                else
                {
                    p = 1 / (1 + Math.Exp(fApB));
                    q = Math.Exp(fApB) / (1 + Math.Exp(fApB));
                }

                double d2 = p * q;
                double d1 = targets[i] - p;

                g1 += si * d1;
                g2 += d1;
                h11 += si * si * d2;
                h22 += d2;
                h21 += si * d2;
            }

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

    private T[] TransformPlattScaling(T[] scores)
    {
        double A = NumOps.ToDouble(_plattA);
        double B = NumOps.ToDouble(_plattB);

        return scores.Select(s =>
        {
            double fApB = NumOps.ToDouble(s) * A + B;
            if (fApB >= 0)
            {
                return NumOps.FromDouble(1 / (1 + Math.Exp(-fApB)));
            }
            else
            {
                double exp = Math.Exp(fApB);
                return NumOps.FromDouble(exp / (1 + exp));
            }
        }).ToArray();
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

        return scores.Select(s =>
        {
            // Binary search for nearest point
            int lo = 0, hi = _isotonicPoints.Count - 1;

            if (NumOps.LessThanOrEquals(s, _isotonicPoints[lo].x)) return _isotonicPoints[lo].y;
            if (NumOps.GreaterThanOrEquals(s, _isotonicPoints[hi].x)) return _isotonicPoints[hi].y;

            while (lo < hi - 1)
            {
                int mid = (lo + hi) / 2;
                if (NumOps.LessThanOrEquals(_isotonicPoints[mid].x, s))
                    lo = mid;
                else
                    hi = mid;
            }

            // Linear interpolation
            T x1 = _isotonicPoints[lo].x;
            T x2 = _isotonicPoints[hi].x;
            T y1 = _isotonicPoints[lo].y;
            T y2 = _isotonicPoints[hi].y;

            T diff = NumOps.Subtract(x2, x1);
            if (NumOps.LessThan(NumOps.Abs(diff), NumOps.FromDouble(1e-10))) return y1;

            return NumOps.Add(y1, NumOps.Divide(NumOps.Multiply(NumOps.Subtract(s, x1), NumOps.Subtract(y2, y1)), diff));
        }).ToArray();
    }

    #endregion

    #region Temperature Scaling

    private void FitTemperatureScaling(T[] scores, T[] labels)
    {
        // Temperature scaling for logits: p = sigmoid(logit / T)
        // If scores are already probabilities, convert to logits first
        // Internal computation in double for numerical stability (exp, log)

        var logits = scores.Select(s =>
        {
            double p = Math.Max(1e-10, Math.Min(1 - 1e-10, NumOps.ToDouble(s)));
            return Math.Log(p / (1 - p));
        }).ToArray();

        var labelsDouble = labels.Select(l => NumOps.ToDouble(l)).ToArray();

        // Optimize temperature to minimize cross-entropy
        double temp = 1.0;
        double lr = _options.LearningRate;

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            double gradient = 0;

            for (int i = 0; i < logits.Length; i++)
            {
                double scaledLogit = logits[i] / temp;
                double p = 1 / (1 + Math.Exp(-scaledLogit));

                // Gradient of cross-entropy w.r.t. T:
                // dL/dT = -(p - y) * logit / T²
                gradient += -(p - labelsDouble[i]) * logits[i] / (temp * temp);
            }

            gradient /= logits.Length;

            if (Math.Abs(gradient) < _options.Tolerance)
            {
                break;
            }

            temp -= lr * gradient;
            temp = Math.Max(0.01, temp);  // Keep T positive
        }

        _temperature = NumOps.FromDouble(temp);
    }

    private T[] TransformTemperatureScaling(T[] scores)
    {
        double temp = NumOps.ToDouble(_temperature);

        return scores.Select(s =>
        {
            double p = Math.Max(1e-10, Math.Min(1 - 1e-10, NumOps.ToDouble(s)));
            double logit = Math.Log(p / (1 - p));
            double scaledLogit = logit / temp;
            return NumOps.FromDouble(1 / (1 + Math.Exp(-scaledLogit)));
        }).ToArray();
    }

    #endregion

    #region Beta Calibration

    private void FitBetaCalibration(T[] scores, T[] labels)
    {
        // Beta calibration: p = 1 / (1 + 1/exp(a*log(s/(1-s)) + b*log(s) + c))
        // Simplified: use logistic regression on transformed features
        // Internal computation in double for numerical stability (exp, log)

        int n = scores.Length;
        var features = new double[n, 3];

        for (int i = 0; i < n; i++)
        {
            double s = Math.Max(1e-6, Math.Min(1 - 1e-6, NumOps.ToDouble(scores[i])));
            features[i, 0] = Math.Log(s / (1 - s));  // logit
            features[i, 1] = Math.Log(s);             // log(s)
            features[i, 2] = 1;                       // intercept
        }

        var labelsDouble = labels.Select(l => NumOps.ToDouble(l)).ToArray();

        // Logistic regression
        double[] w = [0, 0, 0];
        double lr = _options.LearningRate;

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            double[] grad = [0, 0, 0];

            for (int i = 0; i < n; i++)
            {
                double z = w[0] * features[i, 0] + w[1] * features[i, 1] + w[2] * features[i, 2];
                double p = 1 / (1 + Math.Exp(-z));

                for (int j = 0; j < 3; j++)
                {
                    grad[j] += (p - labelsDouble[i]) * features[i, j];
                }
            }

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

    private T[] TransformBetaCalibration(T[] scores)
    {
        double a = NumOps.ToDouble(_betaA);
        double b = NumOps.ToDouble(_betaB);
        double c = NumOps.ToDouble(_betaC);

        return scores.Select(s =>
        {
            double sClamped = Math.Max(1e-6, Math.Min(1 - 1e-6, NumOps.ToDouble(s)));
            double z = a * Math.Log(sClamped / (1 - sClamped)) + b * Math.Log(sClamped) + c;
            return NumOps.FromDouble(1 / (1 + Math.Exp(-z)));
        }).ToArray();
    }

    #endregion

    #region Histogram Binning

    private void FitHistogramBinning(T[] scores, T[] labels)
    {
        int numBins = _options.NumBins;
        _binEdges = new T[numBins + 1];
        _binProbabilities = new T[numBins];

        for (int i = 0; i <= numBins; i++)
        {
            _binEdges[i] = NumOps.FromDouble((double)i / numBins);
        }

        for (int b = 0; b < numBins; b++)
        {
            // Include scores == 1.0 in the last bin
            var binIndices = Enumerable.Range(0, scores.Length)
                .Where(i => NumOps.GreaterThanOrEquals(scores[i], _binEdges[b]) && (b == numBins - 1 ? NumOps.LessThanOrEquals(scores[i], _binEdges[b + 1]) : NumOps.LessThan(scores[i], _binEdges[b + 1])))
                .ToList();

            if (binIndices.Count > 0)
            {
                T sum = NumOps.Zero;
                foreach (var i in binIndices)
                {
                    sum = NumOps.Add(sum, labels[i]);
                }
                _binProbabilities[b] = NumOps.Divide(sum, NumOps.FromDouble(binIndices.Count));
            }
            else
            {
                _binProbabilities[b] = NumOps.Divide(NumOps.Add(_binEdges[b], _binEdges[b + 1]), NumOps.FromDouble(2));
            }
        }
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

        // Convert scores/labels to double for BIC computation
        var scoresDouble = scores.Select(s => NumOps.ToDouble(s)).ToArray();
        var labelsDouble = labels.Select(l => NumOps.ToDouble(l)).ToArray();

        // Sort once and co-sort labels to avoid re-sorting per candidate numBins
        var sortedIndices = Enumerable.Range(0, n).OrderBy(i => scoresDouble[i]).ToArray();
        var sortedScores = sortedIndices.Select(i => scoresDouble[i]).ToArray();
        var sortedLabels = sortedIndices.Select(i => labelsDouble[i]).ToArray();

        for (int numBins = 2; numBins <= maxBins; numBins++)
        {
            // Create equal-frequency (quantile) bin edges from pre-sorted data
            int binSize = n / numBins;

            var edges = new double[numBins + 1];
            edges[0] = 0.0;
            edges[numBins] = 1.0;
            for (int b = 1; b < numBins; b++)
            {
                int idx = Math.Min(b * binSize, n - 1);
                edges[b] = sortedScores[idx];
            }

            // Compute bin statistics using sorted data with binary search for bin boundaries
            double logLikelihood = 0;
            var probs = new double[numBins];

            // Use sorted order to partition samples into bins in O(n) total
            int sampleIdx = 0;
            for (int b = 0; b < numBins; b++)
            {
                int count = 0;
                double positiveCount = 0;
                double upperEdge = edges[b + 1];
                bool isLastBin = b == numBins - 1;

                while (sampleIdx < n)
                {
                    double score = sortedScores[sampleIdx];
                    bool inBin = isLastBin ? score <= upperEdge : score < upperEdge;
                    if (!inBin) break;
                    count++;
                    positiveCount += sortedLabels[sampleIdx];
                    sampleIdx++;
                }

                if (count == 0)
                {
                    probs[b] = (edges[b] + edges[b + 1]) / 2.0;
                    continue;
                }

                double p = positiveCount / count;
                p = Math.Max(1e-10, Math.Min(1 - 1e-10, p));
                probs[b] = p;

                // Binomial log-likelihood for this bin
                logLikelihood += positiveCount * Math.Log(p) + (count - positiveCount) * Math.Log(1 - p);
            }

            // BIC = log-likelihood - (k/2) * log(n), where k = numBins (one parameter per bin)
            double bic = logLikelihood - (numBins / 2.0) * Math.Log(n);

            if (bic > bestBic)
            {
                bestBic = bic;
                bestEdges = edges.Select(e => NumOps.FromDouble(e)).ToArray();
                bestProbs = probs.Select(p => NumOps.FromDouble(p)).ToArray();
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

    private T[] TransformHistogramBinning(T[] scores)
    {
        if (_binEdges == null || _binProbabilities == null)
        {
            return scores;
        }

        return scores.Select(s =>
        {
            for (int b = 0; b < _binProbabilities.Length; b++)
            {
                if (NumOps.GreaterThanOrEquals(s, _binEdges[b]) && NumOps.LessThan(s, _binEdges[b + 1]))
                {
                    return _binProbabilities[b];
                }
            }
            return _binProbabilities[^1];
        }).ToArray();
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
        var calibrated = new T[scores.Length];
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
                T p0 = TransformIsotonicRegression([scores[i]])[0];

                // Augment calibration set with test point labeled 1
                var scores1 = _vennCalibrationScores.Append(scores[i]).ToArray();
                var labels1 = _vennCalibrationLabels.Append(one).ToArray();
                FitIsotonicRegression(scores1, labels1);
                T p1 = TransformIsotonicRegression([scores[i]])[0];

                // Combine to get point estimate
                T denom = NumOps.Add(NumOps.Subtract(one, p0), p1);
                double denomDouble = NumOps.ToDouble(denom);
                calibrated[i] = denomDouble > 1e-10 ? NumOps.Divide(p1, denom) : NumOps.FromDouble(0.5);
            }
        }
        finally
        {
            // Always restore the original isotonic state
            _isotonicPoints = savedIsotonicPoints;
        }

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
