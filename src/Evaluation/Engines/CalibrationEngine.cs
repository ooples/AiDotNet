using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Options;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

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
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var probs = new Tensor<T>(probabilities, new[] { n }).Cast<double>();
        var actualTensor = new Tensor<T>(actuals, new[] { n });

        double ece = 0;
        double mce = 0;
        if (n > 0 && numBins > 0)
        {
            // Create bins: bin i holds i/numBins <= p < (i+1)/numBins (the last bin also holds p == 1)
            var positive = EvaluationTensorOps<T>.GreaterOrEqualMask(actualTensor, NumOps.FromDouble(0.5)).Cast<double>();
            var (counts, meanProbs, frequencies) = EvaluationBinOps.Averages(probs, positive, numBins);
            var gaps = engine.TensorAbs(engine.TensorSubtract(meanProbs, frequencies));

            // Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
            ece = engine.TensorSum(engine.TensorMultiply(counts, gaps)) / n;
            mce = engine.TensorMaxValue(gaps);

            var countValues = counts.Cast<int>().ToArray();
            var meanValues = meanProbs.Cast<T>().ToArray();
            var frequencyValues = frequencies.Cast<T>().ToArray();
            for (int i = 0; i < numBins; i++)
            {
                if (countValues[i] > 0)
                {
                    bins.Add(new CalibrationBin<T>
                    {
                        BinStart = (double)i / numBins,
                        BinEnd = (double)(i + 1) / numBins,
                        MeanPredictedProbability = meanValues[i],
                        ObservedFrequency = frequencyValues[i],
                        SampleCount = countValues[i]
                    });
                }
            }
        }

        // Calculate Brier Score
        var actualIsOne = EvaluationTensorOps<T>.EqualMask(actualTensor, NumOps.One);
        double brier = engine.TensorSumOfSquares(engine.TensorSubtract(probs, actualIsOne.Cast<double>())) / n;

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

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var probs = new Tensor<T>(probabilities, new[] { n }).Cast<double>();
        var targets = EvaluationTensorOps<T>.EqualMask(new Tensor<T>(actuals, new[] { n }), NumOps.One).Cast<double>();

        // Fit sigmoid: P_calibrated = 1 / (1 + exp(A * P_uncalibrated + B))
        // Using simple gradient descent; A and B are one-element tensors broadcast over the samples
        var shape = new[] { n };
        var a = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var b = new Tensor<double>(new[] { 1 });
        const double learningRate = 0.1;
        double count = n;
        int maxIterations = 1000;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            var z = engine.TensorAdd(
                engine.TensorMultiply(probs, engine.TensorBroadcastTo(a, shape)), engine.TensorBroadcastTo(b, shape));
            var error = engine.TensorSubtract(engine.TensorSigmoid(z), targets);

            var gradA = engine.ReduceSum(engine.TensorMultiply(error, probs), new[] { 0 }, true);
            var gradB = engine.ReduceSum(error, new[] { 0 }, true);

            a = engine.TensorSubtract(a, engine.TensorDivideScalar(engine.TensorMultiplyScalar(gradA, learningRate), count));
            b = engine.TensorSubtract(b, engine.TensorDivideScalar(engine.TensorMultiplyScalar(gradB, learningRate), count));
        }

        // Apply calibration
        var calibrated = engine.TensorSigmoid(engine.TensorAdd(
            engine.TensorMultiply(probs, engine.TensorBroadcastTo(a, shape)), engine.TensorBroadcastTo(b, shape))).Cast<T>().ToArray();

        return (calibrated, a[0], b[0]);
    }

    /// <summary>
    /// Applies isotonic regression calibration.
    /// </summary>
    /// <remarks>
    /// Fits the least-squares non-decreasing sequence to the 0/1 labels (actual == 1) ordered by probability, using
    /// the pool-adjacent-violators fit shared with <see cref="Calibration.ProbabilityCalibrator{T}"/>, and returns the
    /// fitted value of each sample in input order (scikit-learn's <c>IsotonicRegression().fit_transform</c> for
    /// distinct probabilities).
    /// </remarks>
    public T[] IsotonicCalibration(T[] probabilities, T[] actuals)
    {
        int n = probabilities.Length;
        if (actuals.Length < n)
            throw new IndexOutOfRangeException("Index was outside the bounds of the array.");
        if (n == 0) return [];

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var labels = EvaluationTensorOps<T>.EqualMask(new Tensor<T>(actuals, new[] { actuals.Length }), NumOps.One).ToArray();

        var calibrator = new Calibration.ProbabilityCalibrator<T>();
        calibrator.FitIsotonicRegression(probabilities, labels);
        var points = calibrator.IsotonicPoints;
        if (points is null || points.Count != n)
            throw new InvalidOperationException("Isotonic fit did not produce one point per sample.");

        // The fit's points are in sorted order (NaN first, ties in input order); scatter them back to input order
        var fitted = new T[n];
        for (int i = 0; i < n; i++)
        {
            fitted[i] = points[i].y;
        }

        var order = EvaluationTensorOps<T>.ArgsortNaNFirst(new Tensor<T>(probabilities, new[] { n }));
        var inverse = engine.TensorArgsort(order);
        return engine.TensorGather(new Tensor<T>(fitted, new[] { n }), inverse, 0).ToArray();
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
    public required T ExpectedCalibrationError { get; init; }

    /// <summary>
    /// Maximum Calibration Error: worst bin's |predicted - observed|.
    /// </summary>
    public required T MaximumCalibrationError { get; init; }

    /// <summary>
    /// Brier Score: mean squared error of probability predictions.
    /// </summary>
    public required T BrierScore { get; init; }

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
    public required T MeanPredictedProbability { get; init; }

    /// <summary>
    /// Observed frequency of positive class in this bin.
    /// </summary>
    public required T ObservedFrequency { get; init; }

    /// <summary>
    /// Number of samples in this bin.
    /// </summary>
    public int SampleCount { get; init; }
}
