using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Options;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Engines;

/// <summary>
/// Engine for analyzing and improving probability calibration of classifiers.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A well-calibrated classifier produces probabilities that match actual outcomes:
/// <list type="bullet">
/// <item>If model says 70% probability, ~70% of such predictions should be positive</item>
/// <item>Many classifiers are NOT well-calibrated (e.g., Random Forest, SVM)</item>
/// <item>Calibration methods: Platt scaling (sigmoid), Isotonic regression</item>
/// </list>
/// </para>
/// <para><b>Why calibration matters:</b>
/// <list type="bullet">
/// <item>Decision making: "Should I act if P > 0.8?" requires calibrated probabilities</item>
/// <item>Ranking: AUC doesn't need calibration, but probability thresholds do</item>
/// <item>Ensemble: Combining models requires comparable probability scales</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CalibrationEngine<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly CalibrationOptions? _options;

    /// <summary>
    /// Initializes the calibration engine.
    /// </summary>
    public CalibrationEngine(CalibrationOptions? options = null)
    {
        _options = options;
    }

    /// <summary>
    /// Analyzes the calibration of a classifier.
    /// </summary>
    /// <param name="probabilities">Predicted probabilities for the positive class.</param>
    /// <param name="actuals">Actual binary labels (0 or 1).</param>
    /// <param name="numBins">Number of bins for calibration curve. Default is 10.</param>
    /// <returns>Calibration analysis results.</returns>
    public CalibrationResult<T> Analyze(T[] probabilities, T[] actuals, int numBins = 10)
    {
        if (probabilities.Length != actuals.Length)
            throw new ArgumentException("Probabilities and actuals must have same length.");

        int n = probabilities.Length;
        var bins = new List<CalibrationBin<T>>();

        // Create bins
        for (int i = 0; i < numBins; i++)
        {
            double binStart = (double)i / numBins;
            double binEnd = (double)(i + 1) / numBins;

            var binProbs = new List<double>();
            var binActuals = new List<int>();

            for (int j = 0; j < n; j++)
            {
                double prob = NumOps.ToDouble(probabilities[j]);
                if (prob >= binStart && (prob < binEnd || (i == numBins - 1 && prob <= binEnd)))
                {
                    binProbs.Add(prob);
                    binActuals.Add(NumOps.ToDouble(actuals[j]) >= 0.5 ? 1 : 0);
                }
            }

            if (binProbs.Count > 0)
            {
                bins.Add(new CalibrationBin<T>
                {
                    BinStart = binStart,
                    BinEnd = binEnd,
                    MeanPredictedProbability = NumOps.FromDouble(binProbs.Average()),
                    ObservedFrequency = NumOps.FromDouble(binActuals.Average()),
                    SampleCount = binProbs.Count
                });
            }
        }

        // Calculate Expected Calibration Error (ECE)
        double ece = 0;
        foreach (var bin in bins)
        {
            double weight = (double)bin.SampleCount / n;
            double diff = Math.Abs(NumOps.ToDouble(bin.MeanPredictedProbability) - NumOps.ToDouble(bin.ObservedFrequency));
            ece += weight * diff;
        }

        // Calculate Maximum Calibration Error (MCE)
        double mce = bins.Count > 0
            ? bins.Max(b => Math.Abs(NumOps.ToDouble(b.MeanPredictedProbability) - NumOps.ToDouble(b.ObservedFrequency)))
            : 0;

        // Calculate Brier Score
        double brier = 0;
        for (int i = 0; i < n; i++)
        {
            double prob = NumOps.ToDouble(probabilities[i]);
            double actual = NumOps.ToDouble(actuals[i]) >= 0.5 ? 1 : 0;
            brier += (prob - actual) * (prob - actual);
        }
        brier /= n;

        return new CalibrationResult<T>
        {
            Bins = bins,
            ExpectedCalibrationError = NumOps.FromDouble(ece),
            MaximumCalibrationError = NumOps.FromDouble(mce),
            BrierScore = NumOps.FromDouble(brier),
            NumSamples = n,
            IsWellCalibrated = ece < 0.05 // Threshold for "well-calibrated"
        };
    }

    /// <summary>
    /// Applies Platt scaling (sigmoid calibration) to probabilities.
    /// </summary>
    /// <param name="probabilities">Uncalibrated probabilities.</param>
    /// <param name="actuals">Actual binary labels.</param>
    /// <returns>Calibrated probabilities and the fitted parameters.</returns>
    public (T[] CalibratedProbabilities, double A, double B) PlattScaling(T[] probabilities, T[] actuals)
    {
        int n = probabilities.Length;

        // Convert to double arrays
        var probs = new double[n];
        var targets = new double[n];
        for (int i = 0; i < n; i++)
        {
            probs[i] = NumOps.ToDouble(probabilities[i]);
            targets[i] = NumOps.ToDouble(actuals[i]) >= 0.5 ? 1 : 0;
        }

        // Fit sigmoid: P_calibrated = 1 / (1 + exp(A * P_uncalibrated + B))
        // Using simple gradient descent
        double a = 1.0, b = 0.0;
        double learningRate = 0.1;
        int maxIterations = 1000;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            double gradA = 0, gradB = 0;

            for (int i = 0; i < n; i++)
            {
                double z = a * probs[i] + b;
                double sigmoid = 1.0 / (1.0 + Math.Exp(-z));
                double error = sigmoid - targets[i];

                gradA += error * probs[i];
                gradB += error;
            }

            a -= learningRate * gradA / n;
            b -= learningRate * gradB / n;
        }

        // Apply calibration
        var calibrated = new T[n];
        for (int i = 0; i < n; i++)
        {
            double z = a * probs[i] + b;
            double sigmoid = 1.0 / (1.0 + Math.Exp(-z));
            calibrated[i] = NumOps.FromDouble(sigmoid);
        }

        return (calibrated, a, b);
    }

    /// <summary>
    /// Applies isotonic regression calibration.
    /// </summary>
    public T[] IsotonicCalibration(T[] probabilities, T[] actuals)
    {
        int n = probabilities.Length;

        // Sort by predicted probability
        var indexed = new (double prob, double actual, int idx)[n];
        for (int i = 0; i < n; i++)
        {
            indexed[i] = (NumOps.ToDouble(probabilities[i]), NumOps.ToDouble(actuals[i]) >= 0.5 ? 1 : 0, i);
        }
        Array.Sort(indexed, (a, b) => a.prob.CompareTo(b.prob));

        // Pool Adjacent Violators Algorithm (PAVA)
        var calibratedValues = new double[n];
        var weights = new int[n];
        for (int i = 0; i < n; i++)
        {
            calibratedValues[i] = indexed[i].actual;
            weights[i] = 1;
        }

        // PAVA: merge adjacent violators
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int i = 0; i < n - 1; i++)
            {
                if (calibratedValues[i] > calibratedValues[i + 1])
                {
                    // Merge
                    double newValue = (calibratedValues[i] * weights[i] + calibratedValues[i + 1] * weights[i + 1])
                                     / (weights[i] + weights[i + 1]);
                    calibratedValues[i] = newValue;
                    calibratedValues[i + 1] = newValue;
                    weights[i] = weights[i] + weights[i + 1];
                    weights[i + 1] = weights[i];
                    changed = true;
                }
            }
        }

        // Map back to original indices
        var result = new T[n];
        for (int i = 0; i < n; i++)
        {
            result[indexed[i].idx] = NumOps.FromDouble(calibratedValues[i]);
        }

        return result;
    }
}

/// <summary>
/// Results from calibration analysis.
/// </summary>
public class CalibrationResult<T>
{
    /// <summary>
    /// Calibration bins with predicted vs observed frequencies.
    /// </summary>
    public List<CalibrationBin<T>> Bins { get; init; } = new();

    /// <summary>
    /// Expected Calibration Error: weighted average of |predicted - observed| across bins.
    /// </summary>
    public T ExpectedCalibrationError { get; init; } = default!;

    /// <summary>
    /// Maximum Calibration Error: worst bin's |predicted - observed|.
    /// </summary>
    public T MaximumCalibrationError { get; init; } = default!;

    /// <summary>
    /// Brier Score: mean squared error of probability predictions.
    /// </summary>
    public T BrierScore { get; init; } = default!;

    /// <summary>
    /// Total number of samples analyzed.
    /// </summary>
    public int NumSamples { get; init; }

    /// <summary>
    /// Whether the classifier is considered well-calibrated (ECE &lt; 0.05).
    /// </summary>
    public bool IsWellCalibrated { get; init; }
}

/// <summary>
/// A single bin in the calibration curve.
/// </summary>
public class CalibrationBin<T>
{
    /// <summary>
    /// Lower bound of the probability bin.
    /// </summary>
    public double BinStart { get; init; }

    /// <summary>
    /// Upper bound of the probability bin.
    /// </summary>
    public double BinEnd { get; init; }

    /// <summary>
    /// Mean predicted probability in this bin.
    /// </summary>
    public T MeanPredictedProbability { get; init; } = default!;

    /// <summary>
    /// Observed frequency of positive class in this bin.
    /// </summary>
    public T ObservedFrequency { get; init; } = default!;

    /// <summary>
    /// Number of samples in this bin.
    /// </summary>
    public int SampleCount { get; init; }
}
