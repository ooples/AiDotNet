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
    private List<(double x, double y)>? _isotonicPoints;

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
    private double[]? _binEdges;
    private double[]? _binProbabilities;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly ProbabilityCalibratorOptions _options;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

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
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
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

        var scoresDouble = scores.Select(s => NumOps.ToDouble(s)).ToArray();
        var labelsDouble = labels.Select(l => NumOps.ToDouble(l)).ToArray();

        switch (_options.CalibratorMethod)
        {
            case ProbabilityCalibrationMethod.PlattScaling:
                FitPlattScaling(scoresDouble, labelsDouble);
                break;

            case ProbabilityCalibrationMethod.IsotonicRegression:
                FitIsotonicRegression(scoresDouble, labelsDouble);
                break;

            case ProbabilityCalibrationMethod.TemperatureScaling:
                FitTemperatureScaling(scoresDouble, labelsDouble);
                break;

            case ProbabilityCalibrationMethod.BetaCalibration:
                FitBetaCalibration(scoresDouble, labelsDouble);
                break;

            case ProbabilityCalibrationMethod.HistogramBinning:
                FitHistogramBinning(scoresDouble, labelsDouble);
                break;

            case ProbabilityCalibrationMethod.BayesianBinning:
                FitBayesianBinning(scoresDouble, labelsDouble);
                break;

            case ProbabilityCalibrationMethod.VennABERS:
                FitVennABERS(scoresDouble, labelsDouble);
                break;

            default:
                FitPlattScaling(scoresDouble, labelsDouble);
                break;
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

        var scoresDouble = scores.Select(s => NumOps.ToDouble(s)).ToArray();
        double[] calibrated;

        switch (_options.CalibratorMethod)
        {
            case ProbabilityCalibrationMethod.PlattScaling:
                calibrated = TransformPlattScaling(scoresDouble);
                break;

            case ProbabilityCalibrationMethod.IsotonicRegression:
                calibrated = TransformIsotonicRegression(scoresDouble);
                break;

            case ProbabilityCalibrationMethod.TemperatureScaling:
                calibrated = TransformTemperatureScaling(scoresDouble);
                break;

            case ProbabilityCalibrationMethod.BetaCalibration:
                calibrated = TransformBetaCalibration(scoresDouble);
                break;

            case ProbabilityCalibrationMethod.HistogramBinning:
            case ProbabilityCalibrationMethod.BayesianBinning:
                calibrated = TransformHistogramBinning(scoresDouble);
                break;

            case ProbabilityCalibrationMethod.VennABERS:
                calibrated = TransformVennABERS(scoresDouble);
                break;

            default:
                calibrated = TransformPlattScaling(scoresDouble);
                break;
        }

        return new Vector<T>(calibrated.Select(c => NumOps.FromDouble(c)).ToArray());
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
        var scoresDouble = scores.Select(s => NumOps.ToDouble(s)).ToArray();
        var labelsDouble = labels.Select(l => NumOps.ToDouble(l)).ToArray();

        int n = scoresDouble.Length;
        double ece = 0;

        for (int b = 0; b < numBins; b++)
        {
            double lower = (double)b / numBins;
            double upper = (double)(b + 1) / numBins;

            var binIndices = Enumerable.Range(0, n)
                .Where(i => scoresDouble[i] >= lower && scoresDouble[i] < upper)
                .ToList();

            if (binIndices.Count > 0)
            {
                double avgProb = binIndices.Average(i => scoresDouble[i]);
                double avgAcc = binIndices.Average(i => labelsDouble[i]);
                double binWeight = (double)binIndices.Count / n;

                ece += binWeight * Math.Abs(avgProb - avgAcc);
            }
        }

        return ece;
    }

    /// <summary>
    /// Computes the Maximum Calibration Error (MCE).
    /// </summary>
    public double ComputeMCE(Vector<T> scores, Vector<T> labels, int numBins = 10)
    {
        var scoresDouble = scores.Select(s => NumOps.ToDouble(s)).ToArray();
        var labelsDouble = labels.Select(l => NumOps.ToDouble(l)).ToArray();

        int n = scoresDouble.Length;
        double mce = 0;

        for (int b = 0; b < numBins; b++)
        {
            double lower = (double)b / numBins;
            double upper = (double)(b + 1) / numBins;

            var binIndices = Enumerable.Range(0, n)
                .Where(i => scoresDouble[i] >= lower && scoresDouble[i] < upper)
                .ToList();

            if (binIndices.Count > 0)
            {
                double avgProb = binIndices.Average(i => scoresDouble[i]);
                double avgAcc = binIndices.Average(i => labelsDouble[i]);

                mce = Math.Max(mce, Math.Abs(avgProb - avgAcc));
            }
        }

        return mce;
    }

    /// <summary>
    /// Computes the Brier Score.
    /// </summary>
    /// <param name="scores">Predicted probabilities.</param>
    /// <param name="labels">True binary labels.</param>
    /// <returns>Brier score (lower is better, 0 is perfect).</returns>
    public double ComputeBrierScore(Vector<T> scores, Vector<T> labels)
    {
        var scoresDouble = scores.Select(s => NumOps.ToDouble(s)).ToArray();
        var labelsDouble = labels.Select(l => NumOps.ToDouble(l)).ToArray();

        double brier = 0;
        for (int i = 0; i < scoresDouble.Length; i++)
        {
            double diff = scoresDouble[i] - labelsDouble[i];
            brier += diff * diff;
        }

        return brier / scoresDouble.Length;
    }

    /// <summary>
    /// Gets a reliability diagram (calibration curve data).
    /// </summary>
    /// <param name="scores">Predicted probabilities.</param>
    /// <param name="labels">True binary labels.</param>
    /// <param name="numBins">Number of bins.</param>
    /// <returns>Tuple of (meanPredicted, fractionPositives, binCounts).</returns>
    public (double[] meanPredicted, double[] fractionPositives, int[] binCounts) GetReliabilityDiagram(
        Vector<T> scores, Vector<T> labels, int numBins = 10)
    {
        var scoresDouble = scores.Select(s => NumOps.ToDouble(s)).ToArray();
        var labelsDouble = labels.Select(l => NumOps.ToDouble(l)).ToArray();

        int n = scoresDouble.Length;
        var meanPredicted = new double[numBins];
        var fractionPositives = new double[numBins];
        var binCounts = new int[numBins];

        for (int b = 0; b < numBins; b++)
        {
            double lower = (double)b / numBins;
            double upper = (double)(b + 1) / numBins;

            var binIndices = Enumerable.Range(0, n)
                .Where(i => scoresDouble[i] >= lower && scoresDouble[i] < upper)
                .ToList();

            binCounts[b] = binIndices.Count;

            if (binIndices.Count > 0)
            {
                meanPredicted[b] = binIndices.Average(i => scoresDouble[i]);
                fractionPositives[b] = binIndices.Average(i => labelsDouble[i]);
            }
            else
            {
                meanPredicted[b] = (lower + upper) / 2;
                fractionPositives[b] = double.NaN;
            }
        }

        return (meanPredicted, fractionPositives, binCounts);
    }

    #region Platt Scaling

    private void FitPlattScaling(double[] scores, double[] labels)
    {
        int n = scores.Length;
        int numPositive = labels.Count(l => l > 0.5);
        int numNegative = n - numPositive;

        // Target values with smoothing
        double hiTarget = (numPositive + 1.0) / (numPositive + 2.0);
        double loTarget = 1.0 / (numNegative + 2.0);

        var targets = labels.Select(l => l > 0.5 ? hiTarget : loTarget).ToArray();

        // Initialize A and B
        double A = 0;
        double B = Math.Log((numNegative + 1.0) / (numPositive + 1.0));

        // Newton-Raphson optimization
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            double g1 = 0, g2 = 0, h11 = 0, h22 = 0, h21 = 0;

            for (int i = 0; i < n; i++)
            {
                double fApB = scores[i] * A + B;
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

                g1 += scores[i] * d1;
                g2 += d1;
                h11 += scores[i] * scores[i] * d2;
                h22 += d2;
                h21 += scores[i] * d2;
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
            if (Math.Abs(det) < 1e-10) det = 1e-10;

            double dA = (g1 * h22 - g2 * h21) / det;
            double dB = (g2 * h11 - g1 * h21) / det;

            A += dA;
            B += dB;
        }

        _plattA = NumOps.FromDouble(A);
        _plattB = NumOps.FromDouble(B);
    }

    private double[] TransformPlattScaling(double[] scores)
    {
        double A = NumOps.ToDouble(_plattA);
        double B = NumOps.ToDouble(_plattB);

        return scores.Select(s =>
        {
            double fApB = s * A + B;
            if (fApB >= 0)
            {
                return 1 / (1 + Math.Exp(-fApB));
            }
            else
            {
                double exp = Math.Exp(fApB);
                return exp / (1 + exp);
            }
        }).ToArray();
    }

    #endregion

    #region Isotonic Regression

    private void FitIsotonicRegression(double[] scores, double[] labels)
    {
        // Pool Adjacent Violators Algorithm (PAVA)
        var sorted = scores.Zip(labels, (s, l) => (score: s, label: l))
            .OrderBy(x => x.score)
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

        // Create calibration points
        _isotonicPoints = [];
        int idx = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double sumWeight = weights[i];
            while (sumWeight > 0)
            {
                _isotonicPoints.Add((sorted[idx].score, y[i]));
                idx++;
                sumWeight--;
            }
        }
    }

    private double[] TransformIsotonicRegression(double[] scores)
    {
        if (_isotonicPoints == null || _isotonicPoints.Count == 0)
        {
            return scores;
        }

        return scores.Select(s =>
        {
            // Binary search for nearest point
            int lo = 0, hi = _isotonicPoints.Count - 1;

            if (s <= _isotonicPoints[lo].x) return _isotonicPoints[lo].y;
            if (s >= _isotonicPoints[hi].x) return _isotonicPoints[hi].y;

            while (lo < hi - 1)
            {
                int mid = (lo + hi) / 2;
                if (_isotonicPoints[mid].x <= s)
                    lo = mid;
                else
                    hi = mid;
            }

            // Linear interpolation
            double x1 = _isotonicPoints[lo].x;
            double x2 = _isotonicPoints[hi].x;
            double y1 = _isotonicPoints[lo].y;
            double y2 = _isotonicPoints[hi].y;

            if (Math.Abs(x2 - x1) < 1e-10) return y1;

            return y1 + (s - x1) * (y2 - y1) / (x2 - x1);
        }).ToArray();
    }

    #endregion

    #region Temperature Scaling

    private void FitTemperatureScaling(double[] scores, double[] labels)
    {
        // Temperature scaling for logits: p = sigmoid(logit / T)
        // If scores are already probabilities, convert to logits first

        var logits = scores.Select(s =>
        {
            double p = Math.Max(1e-10, Math.Min(1 - 1e-10, s));
            return Math.Log(p / (1 - p));
        }).ToArray();

        // Optimize temperature to minimize cross-entropy
        double T = 1.0;
        double lr = _options.LearningRate;

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            double gradient = 0;

            for (int i = 0; i < logits.Length; i++)
            {
                double scaledLogit = logits[i] / T;
                double p = 1 / (1 + Math.Exp(-scaledLogit));

                // Gradient of cross-entropy w.r.t. T
                gradient += (p - labels[i]) * logits[i] / (T * T);
            }

            gradient /= logits.Length;

            if (Math.Abs(gradient) < _options.Tolerance)
            {
                break;
            }

            T -= lr * gradient;
            T = Math.Max(0.01, T);  // Keep T positive
        }

        _temperature = NumOps.FromDouble(T);
    }

    private double[] TransformTemperatureScaling(double[] scores)
    {
        double T = NumOps.ToDouble(_temperature);

        return scores.Select(s =>
        {
            double p = Math.Max(1e-10, Math.Min(1 - 1e-10, s));
            double logit = Math.Log(p / (1 - p));
            double scaledLogit = logit / T;
            return 1 / (1 + Math.Exp(-scaledLogit));
        }).ToArray();
    }

    #endregion

    #region Beta Calibration

    private void FitBetaCalibration(double[] scores, double[] labels)
    {
        // Beta calibration: p = 1 / (1 + 1/exp(a*log(s/(1-s)) + b*log(s) + c))
        // Simplified: use logistic regression on transformed features

        int n = scores.Length;
        var features = new double[n, 3];

        for (int i = 0; i < n; i++)
        {
            double s = Math.Max(1e-6, Math.Min(1 - 1e-6, scores[i]));
            features[i, 0] = Math.Log(s / (1 - s));  // logit
            features[i, 1] = Math.Log(s);             // log(s)
            features[i, 2] = 1;                       // intercept
        }

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
                    grad[j] += (p - labels[i]) * features[i, j];
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

    private double[] TransformBetaCalibration(double[] scores)
    {
        double a = NumOps.ToDouble(_betaA);
        double b = NumOps.ToDouble(_betaB);
        double c = NumOps.ToDouble(_betaC);

        return scores.Select(s =>
        {
            double sClamped = Math.Max(1e-6, Math.Min(1 - 1e-6, s));
            double z = a * Math.Log(sClamped / (1 - sClamped)) + b * Math.Log(sClamped) + c;
            return 1 / (1 + Math.Exp(-z));
        }).ToArray();
    }

    #endregion

    #region Histogram Binning

    private void FitHistogramBinning(double[] scores, double[] labels)
    {
        int numBins = _options.NumBins;
        _binEdges = new double[numBins + 1];
        _binProbabilities = new double[numBins];

        for (int i = 0; i <= numBins; i++)
        {
            _binEdges[i] = (double)i / numBins;
        }

        for (int b = 0; b < numBins; b++)
        {
            var binIndices = Enumerable.Range(0, scores.Length)
                .Where(i => scores[i] >= _binEdges[b] && scores[i] < _binEdges[b + 1])
                .ToList();

            if (binIndices.Count > 0)
            {
                _binProbabilities[b] = binIndices.Average(i => labels[i]);
            }
            else
            {
                _binProbabilities[b] = (_binEdges[b] + _binEdges[b + 1]) / 2;
            }
        }
    }

    private void FitBayesianBinning(double[] scores, double[] labels)
    {
        // Simplified: use adaptive bin widths based on data density
        // Sort scores and create bins with roughly equal counts
        var sorted = scores.OrderBy(s => s).ToArray();
        int n = sorted.Length;
        int numBins = _options.NumBins;
        int binSize = n / numBins;

        _binEdges = new double[numBins + 1];
        _binEdges[0] = 0;
        _binEdges[numBins] = 1;

        for (int b = 1; b < numBins; b++)
        {
            int idx = Math.Min(b * binSize, n - 1);
            _binEdges[b] = sorted[idx];
        }

        _binProbabilities = new double[numBins];
        for (int b = 0; b < numBins; b++)
        {
            var binIndices = Enumerable.Range(0, scores.Length)
                .Where(i => scores[i] >= _binEdges[b] && scores[i] < _binEdges[b + 1])
                .ToList();

            if (binIndices.Count > 0)
            {
                _binProbabilities[b] = binIndices.Average(i => labels[i]);
            }
            else
            {
                _binProbabilities[b] = (_binEdges[b] + _binEdges[b + 1]) / 2;
            }
        }
    }

    private double[] TransformHistogramBinning(double[] scores)
    {
        if (_binEdges == null || _binProbabilities == null)
        {
            return scores;
        }

        return scores.Select(s =>
        {
            for (int b = 0; b < _binProbabilities.Length; b++)
            {
                if (s >= _binEdges![b] && s < _binEdges[b + 1])
                {
                    return _binProbabilities[b];
                }
            }
            return _binProbabilities[^1];
        }).ToArray();
    }

    #endregion

    #region Venn-ABERS

    private void FitVennABERS(double[] scores, double[] labels)
    {
        // Venn-ABERS fits isotonic regression twice:
        // once assuming new point is positive, once assuming negative
        // For simplicity, we just fit isotonic regression here
        FitIsotonicRegression(scores, labels);
    }

    private double[] TransformVennABERS(double[] scores)
    {
        // In full implementation, this would return probability intervals
        // For now, return point estimates from isotonic regression
        return TransformIsotonicRegression(scores);
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
