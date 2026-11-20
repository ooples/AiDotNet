namespace AiDotNet.UncertaintyQuantification.Calibration;

using System.Linq;

/// <summary>
/// Computes the Expected Calibration Error (ECE) metric for evaluating probability calibration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Expected Calibration Error (ECE) measures how well a model's predicted
/// probabilities match actual outcomes.
///
/// Imagine a weather forecaster who predicts "70% chance of rain":
/// - If it rains 70% of the time when they make this prediction, they're well-calibrated (ECE near 0)
/// - If it only rains 50% of the time, they're overconfident (ECE is higher)
///
/// How ECE works:
/// 1. Group all predictions into bins by confidence (e.g., 0-10%, 10-20%, ..., 90-100%)
/// 2. For each bin, compute:
///    - Average predicted probability
///    - Actual accuracy (fraction of correct predictions)
/// 3. ECE is the weighted average of the absolute differences
///
/// A well-calibrated model should have ECE &lt; 0.05 (5%).
///
/// This is the gold standard metric for evaluating uncertainty quantification in classification tasks.
/// </para>
/// </remarks>
public class ExpectedCalibrationError<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numBins;

    /// <summary>
    /// Initializes a new instance of the ExpectedCalibrationError class.
    /// </summary>
    /// <param name="numBins">Number of bins for grouping predictions (default: 10).</param>
    /// <remarks>
    /// <b>For Beginners:</b> More bins give finer-grained calibration analysis but require more data.
    /// 10-15 bins is standard for most applications.
    /// </remarks>
    public ExpectedCalibrationError(int numBins = 10)
    {
        if (numBins < 1)
            throw new ArgumentException("Number of bins must be at least 1", nameof(numBins));

        _numOps = MathHelper.GetNumericOperations<T>();
        _numBins = numBins;
    }

    /// <summary>
    /// Computes the Expected Calibration Error for predicted probabilities.
    /// </summary>
    /// <param name="probabilities">Predicted probabilities (values between 0 and 1).</param>
    /// <param name="predictions">Predicted class labels.</param>
    /// <param name="trueLabels">True class labels.</param>
    /// <returns>The ECE value (lower is better, 0 is perfect calibration).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Pass in your model's predicted probabilities and the true labels.
    /// The method returns a score where:
    /// - 0.0 = Perfect calibration
    /// - &lt; 0.05 = Good calibration
    /// - 0.05-0.10 = Moderate calibration
    /// - > 0.10 = Poor calibration (needs calibration methods like temperature scaling)
    /// </remarks>
    public T Compute(Vector<T> probabilities, Vector<int> predictions, Vector<int> trueLabels)
    {
        if (probabilities.Length != predictions.Length || probabilities.Length != trueLabels.Length)
            throw new ArgumentException("All input vectors must have the same length");

        if (probabilities.Length == 0)
            return _numOps.Zero;

        var engine = AiDotNetEngine.Current;
        var (_, correctCounts, sumConfidence) = ComputeBinStatistics(probabilities, predictions, trueLabels);

        // ECE = sum_m (n_m / n) |acc_m - conf_m| = sum_m |correct_m - sumConfidence_m| / n.
        // Empty bins contribute zero on their own.
        var gap = engine.TensorAbs(engine.TensorSubtract(correctCounts.Cast<T>(), sumConfidence));
        return _numOps.Divide(engine.TensorSum(gap), _numOps.FromDouble(probabilities.Length));
    }

    /// <summary>
    /// Gets reliability diagram data for visualization.
    /// </summary>
    /// <param name="probabilities">Predicted probabilities.</param>
    /// <param name="predictions">Predicted class labels.</param>
    /// <param name="trueLabels">True class labels.</param>
    /// <returns>One entry per non-empty bin, in bin order: (average confidence, accuracy, sample count).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> A reliability diagram plots confidence vs accuracy.
    /// Perfect calibration would show all points on the diagonal line (confidence = accuracy).
    /// Points below the diagonal indicate overconfidence; points above indicate underconfidence.
    /// </remarks>
    public List<(T confidence, T accuracy, int count)> GetReliabilityDiagram(
        Vector<T> probabilities, Vector<int> predictions, Vector<int> trueLabels)
    {
        if (probabilities.Length != predictions.Length || probabilities.Length != trueLabels.Length)
            throw new ArgumentException("All input vectors must have the same length");

        if (probabilities.Length == 0)
            return new List<(T confidence, T accuracy, int count)>();

        var engine = AiDotNetEngine.Current;
        var (counts, correctCounts, sumConfidence) = ComputeBinStatistics(probabilities, predictions, trueLabels);

        // Keep the non-empty bins; dividing by max(count, 1) leaves the empty ones finite before they are dropped.
        var occupied = engine.TensorEqScalar(engine.TensorGreaterThan(counts, 0), 1);
        var countsT = engine.TensorClampMin(counts, 1).Cast<T>();
        var confidence = engine.TensorMaskedSelect(engine.TensorDivide(sumConfidence, countsT), occupied).ToArray();
        var accuracy = engine.TensorMaskedSelect(engine.TensorDivide(correctCounts.Cast<T>(), countsT), occupied).ToArray();
        var binCounts = engine.TensorMaskedSelect(counts, occupied).ToArray();

        return confidence
            .Zip(accuracy, (c, a) => (c, a))
            .Zip(binCounts, (ca, n) => (confidence: ca.c, accuracy: ca.a, count: n))
            .ToList();
    }

    /// <summary>
    /// Per-bin sample counts, correct counts and confidence sums.
    /// </summary>
    /// <remarks>
    /// Bin m (1-based) holds confidences in ((m-1)/M, m/M] (Naeini et al. 2015; Guo et al. 2017, Sec. 2),
    /// i.e. index ceil(p M) - 1, clamped to [0, M-1] so p = 0 joins the first bin. The product p M rounds
    /// (0.07 * 100 = 7.000000000000001 in double, whose ceiling is 8), so the candidate index is then checked
    /// against the boundaries k / M themselves, the values np.linspace(0, 1, M + 1) holds, and moved one bin
    /// down where p &lt;= its lower boundary or up where p &gt; its upper one.
    /// The per-bin sums are segmented sums over the bin index, so memory is linear in the samples plus the
    /// bins, with no [samples x bins] membership. Counts stay int so they are exact for every T.
    /// </remarks>
    private (Tensor<int> Counts, Tensor<int> CorrectCounts, Tensor<T> SumConfidence) ComputeBinStatistics(
        Vector<T> probabilities, Vector<int> predictions, Vector<int> trueLabels)
    {
        var engine = AiDotNetEngine.Current;
        int n = probabilities.Length;
        var binCount = _numOps.FromDouble(_numBins);
        var lastBin = _numOps.FromDouble(_numBins - 1);

        var confidence = Tensor<T>.FromVector(probabilities, new[] { n });
        var candidate = engine.TensorClamp(
            engine.TensorSubtractScalar(engine.TensorCeiling(engine.TensorMultiplyScalar(confidence, binCount)), _numOps.One),
            _numOps.Zero, lastBin);
        var lower = engine.TensorDivideScalar(candidate, binCount);
        var upper = engine.TensorDivideScalar(engine.TensorAddScalar(candidate, _numOps.One), binCount);
        var atOrBelowLower = engine.ScalarMinusTensor(_numOps.One, engine.TensorGreaterThan(confidence, lower));
        var aboveUpper = engine.TensorGreaterThan(confidence, upper);
        var binIndex = engine.TensorClamp(
            engine.TensorAdd(engine.TensorSubtract(candidate, atOrBelowLower), aboveUpper), _numOps.Zero, lastBin);
        var bins = binIndex.Cast<int>();

        var counts = engine.TensorBinCount(bins, _numBins);                                          // [M]
        var correct = engine.TensorEquals(
            Tensor<int>.FromVector(predictions, new[] { n }), Tensor<int>.FromVector(trueLabels, new[] { n }));
        var correctCounts = engine.ScatterAdd(correct, bins, 0, _numBins);                           // [M]
        var sumConfidence = engine.ScatterAdd(confidence, bins, 0, _numBins);                        // [M]

        return (counts, correctCounts, sumConfidence);
    }
}
