using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using Hampel identifier (median-based outlier detection).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Hampel identifier is a robust outlier detection method that uses
/// a sliding window to compute local median and MAD. It's particularly effective for time series
/// data where local context matters.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. For each point, compute median and MAD in a local window
/// 2. Compute the deviation from local median in MAD units
/// 3. Points exceeding the threshold are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Time series with local patterns
/// - When global statistics don't capture local anomalies
/// - As a robust alternative to moving average methods
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Window size: 7 (3 on each side)
/// - Threshold: 3 MAD units
/// - Scale factor: 1.4826
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Hampel, F.R. (1974). "The Influence Curve and its Role in Robust Estimation."
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("The Influence Curve and its Role in Robust Estimation", "https://doi.org/10.1080/01621459.1974.10482962")]
public class HampelDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _windowSize;
    private readonly double _madThreshold;
    private readonly double _scaleFactor;

    /// <summary>
    /// Gets the window size.
    /// </summary>
    public int WindowSize => _windowSize;

    /// <summary>
    /// Gets the threshold in MAD units.
    /// </summary>
    public double ThresholdMAD => _madThreshold;

    /// <summary>
    /// Creates a new Hampel anomaly detector.
    /// </summary>
    /// <param name="windowSize">Size of the local window. Default is 7.</param>
    /// <param name="threshold">Threshold in MAD units. Default is 3.</param>
    /// <param name="scaleFactor">MAD scale factor. Default is 1.4826.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public HampelDetector(int windowSize = 7, double threshold = 3, double scaleFactor = 1.4826,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (windowSize < 3)
        {
            throw new ArgumentOutOfRangeException(nameof(windowSize),
                "WindowSize must be at least 3. Recommended is 7.");
        }

        if (threshold <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be positive. Recommended is 3.");
        }

        _windowSize = windowSize;
        _madThreshold = threshold;
        _scaleFactor = scaleFactor;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        int d = X.Columns;
        int halfWindow = _windowSize / 2;

        T eps = NumOps.FromDouble(1e-10);
        T scaleT = NumOps.FromDouble(_scaleFactor);

        var engine = Engine;
        var data = AnomalyTensorOps.ToTensor(X);
        var lengths = AnomalyTensorOps.SlidingWindowLengths(n, halfWindow, halfWindow);
        var scores = Tensor<T>.CreateZeros(new[] { n });

        for (int j = 0; j < d; j++)
        {
            // Local window [i - halfWindow, i + halfWindow] of this feature, truncated at the ends
            var series = engine.TensorSliceAxis(data, 1, j);
            var windows = AnomalyTensorOps.SlidingWindows(series, halfWindow, halfWindow, out var mask);

            // Local median
            var localMedian = AnomalyTensorOps.MaskedRowMedians(windows, mask, lengths);

            // Local MAD, scaled and floored at eps
            var absDeviations = engine.TensorAbs(
                engine.TensorSubtract(windows, AnomalyTensorOps.BroadcastColumn(localMedian, windows.Shape[1])));
            var localMad = AnomalyTensorOps.MaskedRowMedians(absDeviations, mask, lengths);
            localMad = engine.TensorMax(engine.TensorMultiplyScalar(localMad, scaleT), eps);

            // Score for this feature; the point's score is the largest one (a NaN score never wins)
            var featureScore = engine.TensorDivide(engine.TensorAbs(engine.TensorSubtract(series, localMedian)), localMad);
            featureScore = engine.TensorMaskedFill(featureScore, engine.TensorIsNan(featureScore), NumOps.Zero);
            scores = engine.TensorMax(scores, featureScore);
        }

        return scores.ToVector();
    }
}
