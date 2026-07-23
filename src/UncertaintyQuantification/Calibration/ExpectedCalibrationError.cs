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
/// A well-calibrated model should have ECE < 0.05 (5%).
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
    /// - < 0.05 = Good calibration
    /// - 0.05-0.10 = Moderate calibration
    /// - > 0.10 = Poor calibration (needs calibration methods like temperature scaling)
    /// </remarks>
    public T Compute(Vector<T> probabilities, Vector<int> predictions, Vector<int> trueLabels)
    {
        if (probabilities.Length != predictions.Length || probabilities.Length != trueLabels.Length)
            throw new ArgumentException("All input vectors must have the same length");

        var bins = CreateBins(probabilities, predictions, trueLabels);
        var ece = _numOps.Zero;
        var totalSamples = _numOps.FromDouble(probabilities.Length);

        foreach (var bin in bins.Where(b => b.Count != 0))
        {
            // Compute average confidence in this bin
            var avgConfidence = _numOps.Divide(bin.SumConfidence, _numOps.FromDouble(bin.Count));

            // Compute accuracy in this bin
            var accuracy = _numOps.Divide(_numOps.FromDouble(bin.CorrectCount), _numOps.FromDouble(bin.Count));

            // |accuracy - confidence|
            var diff = _numOps.Abs(_numOps.Subtract(accuracy, avgConfidence));

            // Weight by proportion of samples in this bin
            var weight = _numOps.Divide(_numOps.FromDouble(bin.Count), totalSamples);

            // Add weighted difference to ECE
            ece = _numOps.Add(ece, _numOps.Multiply(weight, diff));
        }

        return ece;
    }

    /// <summary>
    /// Computes a reliability diagram showing calibration across bins.
    /// </summary>
    /// <param name="probabilities">Predicted probabilities.</param>
    /// <param name="predictions">Predicted class labels.</param>
    /// <param name="trueLabels">True class labels.</param>
    /// <returns>A list of (confidence, accuracy, count) tuples for each bin.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This returns data for plotting a reliability diagram,
    /// which visualizes how well-calibrated your model is across different confidence levels.
    /// In a perfect model, the points would lie on a diagonal line (confidence = accuracy).
    /// </remarks>
    public List<(double confidence, double accuracy, int count)> GetReliabilityDiagram(
        Vector<T> probabilities, Vector<int> predictions, Vector<int> trueLabels)
    {
        var bins = CreateBins(probabilities, predictions, trueLabels);
        var diagram = new List<(double confidence, double accuracy, int count)>();

        foreach (var bin in bins.Where(b => b.Count != 0))
        {
            var avgConfidence = Convert.ToDouble(bin.SumConfidence) / bin.Count;
            var accuracy = (double)bin.CorrectCount / bin.Count;

            diagram.Add((avgConfidence, accuracy, bin.Count));
        }

        return diagram;
    }

    /// <summary>
    /// Creates bins for grouping predictions by confidence level.
    /// </summary>
    private List<CalibrationBin<T>> CreateBins(Vector<T> probabilities, Vector<int> predictions, Vector<int> trueLabels)
    {
        var bins = new List<CalibrationBin<T>>();
        for (int i = 0; i < _numBins; i++)
        {
            bins.Add(new CalibrationBin<T>(_numOps));
        }

        for (int i = 0; i < probabilities.Length; i++)
        {
            var prob = probabilities[i];
            var probDouble = Convert.ToDouble(prob);

            // Determine which bin this prediction belongs to
            int binIndex = (int)(probDouble * _numBins);
            if (binIndex >= _numBins) binIndex = _numBins - 1; // Handle prob = 1.0
            if (binIndex < 0) binIndex = 0; // Handle prob = 0.0

            // Add to bin
            var isCorrect = predictions[i] == trueLabels[i];
            bins[binIndex].Add(prob, isCorrect);
        }

        return bins;
    }
}

/// <summary>
/// Represents a bin for calibration analysis.
/// </summary>
internal class CalibrationBin<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Sum of the confidence values added to this bin.
    /// </summary>
    public T SumConfidence { get; private set; }

    /// <summary>
    /// Number of samples added to this bin.
    /// </summary>
    public int Count { get; private set; }

    /// <summary>
    /// Number of correctly predicted samples in this bin.
    /// </summary>
    public int CorrectCount { get; private set; }

    /// <summary>
    /// Initializes a new calibration bin.
    /// </summary>
    /// <param name="numOps">Numeric operations implementation for <typeparamref name="T"/>.</param>
    public CalibrationBin(INumericOperations<T> numOps)
    {
        _numOps = numOps;
        SumConfidence = numOps.Zero;
        Count = 0;
        CorrectCount = 0;
    }

    /// <summary>
    /// Adds a prediction outcome to the bin.
    /// </summary>
    /// <param name="confidence">The predicted confidence.</param>
    /// <param name="isCorrect">Whether the prediction was correct.</param>
    public void Add(T confidence, bool isCorrect)
    {
        SumConfidence = _numOps.Add(SumConfidence, confidence);
        Count++;
        if (isCorrect) CorrectCount++;
    }
}
