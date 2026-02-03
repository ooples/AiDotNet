namespace AiDotNet.DriftDetection;

/// <summary>
/// Page-Hinkley test for concept drift detection using cumulative sum analysis.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The Page-Hinkley test is a sequential analysis technique that
/// monitors a cumulative sum of deviations from the mean. When the cumulative sum exceeds a
/// threshold, drift is detected. It's particularly good at detecting changes in the mean of a stream.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Calculate the running mean x̄ of observations</item>
/// <item>For each new value x, update cumulative sum: m = m + (x - x̄ - δ)</item>
/// <item>Track the minimum value M of the cumulative sum</item>
/// <item>When (m - M) > λ, drift is detected</item>
/// </list>
/// </para>
///
/// <para><b>Parameters:</b>
/// <list type="bullet">
/// <item><b>delta (δ):</b> Magnitude of allowed changes - helps filter noise (default: 0.005)</item>
/// <item><b>lambda (λ):</b> Detection threshold - higher = less sensitive (default: 50)</item>
/// </list>
/// </para>
///
/// <para><b>Key insight:</b> The cumulative sum tracks deviations from expected behavior.
/// Normal fluctuations cancel out over time, but a true change causes monotonic increase.</para>
///
/// <para><b>Advantages:</b>
/// <list type="bullet">
/// <item>Based on solid statistical foundation (sequential analysis)</item>
/// <item>Low computational cost</item>
/// <item>Works well for detecting changes in mean</item>
/// <item>Can be adapted for two-sided detection (increases and decreases)</item>
/// </list>
/// </para>
///
/// <para><b>Limitations:</b>
/// <list type="bullet">
/// <item>Requires tuning delta and lambda parameters</item>
/// <item>Primarily detects shifts in mean, not other distribution changes</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Page, E. S. (1954). "Continuous Inspection Schemes"</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PageHinkleyDriftDetector<T> : DriftDetectorBase<T>
{
    private readonly double _delta;
    private readonly double _lambda;
    private readonly double _warningFraction;
    private readonly bool _twoSided;

    /// <summary>
    /// Running sum of observations.
    /// </summary>
    private double _sum;

    /// <summary>
    /// Cumulative sum statistic (for increase detection).
    /// </summary>
    private double _cumulativeSumUp;

    /// <summary>
    /// Cumulative sum statistic (for decrease detection, in two-sided mode).
    /// </summary>
    private double _cumulativeSumDown;

    /// <summary>
    /// Minimum value of cumulative sum (for increase detection).
    /// </summary>
    private double _minCumulativeSumUp;

    /// <summary>
    /// Maximum value of cumulative sum (for decrease detection).
    /// </summary>
    private double _maxCumulativeSumDown;

    /// <summary>
    /// Gets whether the detector is in warning zone.
    /// </summary>
    public new bool IsInWarning { get; private set; }

    /// <summary>
    /// Creates a new Page-Hinkley drift detector.
    /// </summary>
    /// <param name="delta">Magnitude threshold for counting changes (default: 0.005).</param>
    /// <param name="lambda">Detection threshold (default: 50).</param>
    /// <param name="warningFraction">Fraction of lambda for warning (default: 0.5).</param>
    /// <param name="twoSided">If true, detects both increases and decreases (default: false).</param>
    /// <param name="minimumObservations">Minimum samples before detection (default: 30).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// <list type="bullet">
    /// <item>delta: Small values detect small changes, large values ignore small fluctuations</item>
    /// <item>lambda: Larger values require more evidence before declaring drift</item>
    /// <item>warningFraction: How much of lambda triggers a warning (e.g., 0.5 = 50%)</item>
    /// <item>twoSided: Set true to detect drift in either direction (mean going up or down)</item>
    /// </list>
    /// </para>
    /// </remarks>
    public PageHinkleyDriftDetector(
        double delta = 0.005,
        double lambda = 50.0,
        double warningFraction = 0.5,
        bool twoSided = false,
        int minimumObservations = 30)
    {
        if (delta < 0)
        {
            throw new ArgumentException("Delta must be non-negative.", nameof(delta));
        }
        if (lambda <= 0)
        {
            throw new ArgumentException("Lambda must be positive.", nameof(lambda));
        }
        if (warningFraction <= 0 || warningFraction >= 1)
        {
            throw new ArgumentException("Warning fraction must be between 0 and 1.", nameof(warningFraction));
        }

        _delta = delta;
        _lambda = lambda;
        _warningFraction = warningFraction;
        _twoSided = twoSided;
        MinimumObservations = minimumObservations;
        Reset();
    }

    /// <summary>
    /// Adds a new observation to the detector.
    /// </summary>
    /// <param name="value">The new observation value.</param>
    /// <returns>True if drift is detected.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Feed this with your metric of interest (e.g., model error,
    /// prediction confidence, or any quantity that might change during drift).</para>
    /// </remarks>
    public override bool AddObservation(T value)
    {
        double x = ToDouble(value);
        ObservationCount++;
        _sum += x;

        // Update running mean
        EstimatedMean = _sum / ObservationCount;

        // Skip drift detection until minimum observations
        if (ObservationCount < MinimumObservations)
        {
            return false;
        }

        // Update cumulative sum for detecting mean increase
        // m_t = m_{t-1} + (x_t - x̄ - δ)
        _cumulativeSumUp += x - EstimatedMean - _delta;

        // Track minimum cumulative sum
        if (_cumulativeSumUp < _minCumulativeSumUp)
        {
            _minCumulativeSumUp = _cumulativeSumUp;
        }

        // Calculate Page-Hinkley statistic (deviation from minimum)
        double phStatisticUp = _cumulativeSumUp - _minCumulativeSumUp;

        double phStatisticDown = 0;
        if (_twoSided)
        {
            // Update cumulative sum for detecting mean decrease
            // m_t = m_{t-1} - (x_t - x̄ - δ)
            _cumulativeSumDown -= x - EstimatedMean - _delta;

            // Track maximum cumulative sum (which represents minimum when negated)
            if (_cumulativeSumDown > _maxCumulativeSumDown)
            {
                _maxCumulativeSumDown = _cumulativeSumDown;
            }

            phStatisticDown = _maxCumulativeSumDown - _cumulativeSumDown;
        }

        // Use the maximum of both directions
        double phStatistic = Math.Max(phStatisticUp, phStatisticDown);

        // Update drift probability
        DriftProbability = Math.Min(1.0, phStatistic / _lambda);

        // Calculate warning threshold
        double warningThreshold = _lambda * _warningFraction;

        // Check for warning
        if (phStatistic > warningThreshold && phStatistic <= _lambda)
        {
            IsInWarning = true;
        }
        else
        {
            IsInWarning = false;
        }

        // Check for drift
        if (phStatistic > _lambda)
        {
            IsInDrift = true;
            return true;
        }

        return false;
    }

    /// <summary>
    /// Resets the detector to its initial state.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        _sum = 0;
        _cumulativeSumUp = 0;
        _cumulativeSumDown = 0;
        _minCumulativeSumUp = double.MaxValue;
        _maxCumulativeSumDown = double.MinValue;
        IsInWarning = false;
    }

    /// <summary>
    /// Gets the current Page-Hinkley statistic for mean increase detection.
    /// </summary>
    public double PageHinkleyStatisticUp => _cumulativeSumUp - _minCumulativeSumUp;

    /// <summary>
    /// Gets the current Page-Hinkley statistic for mean decrease detection (if two-sided).
    /// </summary>
    public double PageHinkleyStatisticDown => _twoSided ? _maxCumulativeSumDown - _cumulativeSumDown : 0;

    /// <summary>
    /// Gets the detection threshold (lambda).
    /// </summary>
    public double Threshold => _lambda;

    /// <summary>
    /// Gets the magnitude threshold (delta).
    /// </summary>
    public double Delta => _delta;

    /// <summary>
    /// Gets whether two-sided detection is enabled.
    /// </summary>
    public bool IsTwoSided => _twoSided;
}
