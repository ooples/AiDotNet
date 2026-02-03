namespace AiDotNet.DriftDetection;

/// <summary>
/// Implements Page-Hinkley Test for concept drift detection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The Page-Hinkley test is a sequential analysis method that
/// detects changes in the mean of a process. Unlike DDM/EDDM which are designed for binary
/// errors, Page-Hinkley works with continuous values (like loss values or accuracy scores).</para>
///
/// <para><b>How Page-Hinkley works:</b>
/// <list type="number">
/// <item>Track the cumulative sum of deviations from the running mean</item>
/// <item>Monitor the difference between cumulative sum and its minimum value</item>
/// <item>Drift is detected when this difference exceeds a threshold (λ)</item>
/// </list>
/// The test accumulates evidence of change over time, making it robust to noise.
/// </para>
///
/// <para><b>Key Parameters:</b>
/// <list type="bullet">
/// <item><b>λ (lambda):</b> Detection threshold - larger values = fewer false alarms but slower detection</item>
/// <item><b>α (alpha):</b> Magnitude of allowed change - helps ignore small fluctuations</item>
/// </list>
/// </para>
///
/// <para><b>Variants:</b>
/// <list type="bullet">
/// <item><b>One-sided (decrease):</b> Detects when values decrease (e.g., accuracy dropping)</item>
/// <item><b>One-sided (increase):</b> Detects when values increase (e.g., loss increasing)</item>
/// <item><b>Two-sided:</b> Detects changes in either direction</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Monitoring continuous metrics (loss, accuracy, scores)</item>
/// <item>When you need to detect changes in mean value</item>
/// <item>When you want control over detection sensitivity</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Page, "Continuous Inspection Schemes" (1954)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PageHinkley<T> : DriftDetectorBase<T>
{
    private double _sum;
    private double _sumMax;
    private double _sumMin;
    private double _runningMean;

    /// <summary>
    /// Detection mode for Page-Hinkley test.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Choose based on what kind of change you want to detect.</para>
    /// </remarks>
    public enum DetectionMode
    {
        /// <summary>Detect decreases in value (e.g., accuracy dropping).</summary>
        DetectDecrease,
        /// <summary>Detect increases in value (e.g., loss increasing).</summary>
        DetectIncrease,
        /// <summary>Detect both increases and decreases.</summary>
        DetectBoth
    }

    /// <summary>
    /// Gets the detection threshold (lambda).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Larger values mean you need more evidence to declare drift,
    /// resulting in fewer false alarms but slower detection. Typical values: 10-100.</para>
    /// </remarks>
    public double Lambda { get; }

    /// <summary>
    /// Gets the tolerance parameter (alpha).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This parameter allows small fluctuations without triggering
    /// drift. Higher values make the detector less sensitive. Typical values: 0.001-0.05.</para>
    /// </remarks>
    public double Alpha { get; }

    /// <summary>
    /// Gets the detection mode.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For monitoring accuracy, use DetectDecrease.
    /// For monitoring loss/errors, use DetectIncrease. Use DetectBoth if unsure.</para>
    /// </remarks>
    public DetectionMode Mode { get; }

    /// <summary>
    /// Creates a new Page-Hinkley drift detector.
    /// </summary>
    /// <param name="lambda">Detection threshold (default: 50).</param>
    /// <param name="alpha">Tolerance for small changes (default: 0.005).</param>
    /// <param name="mode">Detection direction (default: DetectBoth).</param>
    /// <param name="minimumObservations">Minimum samples before detection (default: 30).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Start with defaults. If too many false alarms, increase lambda.
    /// If detection is too slow, decrease lambda. Adjust alpha if small natural fluctuations
    /// cause problems.</para>
    /// </remarks>
    public PageHinkley(
        double lambda = 50,
        double alpha = 0.005,
        DetectionMode mode = DetectionMode.DetectBoth,
        int minimumObservations = 30)
    {
        if (lambda <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(lambda), "Lambda must be positive.");
        }

        if (alpha < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be non-negative.");
        }

        Lambda = lambda;
        Alpha = alpha;
        Mode = mode;
        MinimumObservations = minimumObservations;

        Reset();
    }

    /// <inheritdoc />
    public override void Reset()
    {
        base.Reset();
        _sum = 0;
        _sumMax = double.MinValue;
        _sumMin = double.MaxValue;
        _runningMean = 0;
    }

    /// <summary>
    /// Adds a new observation to the Page-Hinkley test.
    /// </summary>
    /// <param name="value">The observation (continuous value like loss or accuracy).</param>
    /// <returns>True if drift was detected.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Pass your metric value (loss, accuracy, score) for each
    /// sample or batch. The detector will track changes in the average value over time.</para>
    /// </remarks>
    public override bool AddObservation(T value)
    {
        double val = ToDouble(value);
        ObservationCount++;

        // Update running mean using Welford's method
        double delta = val - _runningMean;
        _runningMean += delta / ObservationCount;

        EstimatedMean = _runningMean;

        // Update cumulative sum
        // For detecting increases: sum up (value - mean - alpha)
        // For detecting decreases: sum up (mean - value - alpha)
        _sum += (val - _runningMean - Alpha);

        // Track min and max of cumulative sum
        _sumMax = Math.Max(_sumMax, _sum);
        _sumMin = Math.Min(_sumMin, _sum);

        // Not enough observations yet
        if (ObservationCount < MinimumObservations)
        {
            return false;
        }

        bool driftDetected = false;

        // Check for drift based on mode
        switch (Mode)
        {
            case DetectionMode.DetectIncrease:
                // Drift if cumulative sum increases significantly above minimum
                if (_sum - _sumMin > Lambda)
                {
                    driftDetected = true;
                }
                DriftProbability = Math.Min(1.0, (_sum - _sumMin) / Lambda);
                break;

            case DetectionMode.DetectDecrease:
                // Drift if cumulative sum decreases significantly below maximum
                if (_sumMax - _sum > Lambda)
                {
                    driftDetected = true;
                }
                DriftProbability = Math.Min(1.0, (_sumMax - _sum) / Lambda);
                break;

            case DetectionMode.DetectBoth:
                // Drift if either direction shows significant change
                double increaseTest = _sum - _sumMin;
                double decreaseTest = _sumMax - _sum;
                double maxTest = Math.Max(increaseTest, decreaseTest);

                if (maxTest > Lambda)
                {
                    driftDetected = true;
                }
                DriftProbability = Math.Min(1.0, maxTest / Lambda);
                break;
        }

        // Set warning at 80% of threshold
        IsInWarning = DriftProbability > 0.8 && !driftDetected;
        IsInDrift = driftDetected;

        return driftDetected;
    }

    /// <summary>
    /// Gets the current cumulative sum value.
    /// </summary>
    /// <returns>The cumulative sum of deviations from mean.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This value accumulates evidence of change. Large positive
    /// or negative values (depending on mode) indicate drift is likely.</para>
    /// </remarks>
    public double GetCumulativeSum()
    {
        return _sum;
    }

    /// <summary>
    /// Gets the current test statistic value.
    /// </summary>
    /// <returns>The Page-Hinkley test statistic.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the value compared against lambda to detect drift.
    /// Track it over time to see how close you are to drift detection.</para>
    /// </remarks>
    public double GetTestStatistic()
    {
        return Mode switch
        {
            DetectionMode.DetectIncrease => _sum - _sumMin,
            DetectionMode.DetectDecrease => _sumMax - _sum,
            DetectionMode.DetectBoth => Math.Max(_sum - _sumMin, _sumMax - _sum),
            _ => 0
        };
    }
}
