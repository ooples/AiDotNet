namespace AiDotNet.DriftDetection;

/// <summary>
/// Drift Detection Method (DDM) for concept drift detection based on error rate monitoring.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> DDM monitors the error rate of a classifier over time. When
/// the error rate increases significantly beyond what was observed during a stable period,
/// drift is detected. DDM is simple, fast, and effective for detecting sudden drift.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Track the error rate p and its standard deviation s = sqrt(p(1-p)/n)</item>
/// <item>Remember the minimum observed value of p + s (the baseline)</item>
/// <item>Warning level: p + s > p_min + s_min + warning_threshold × s_min</item>
/// <item>Drift level: p + s > p_min + s_min + drift_threshold × s_min</item>
/// </list>
/// </para>
///
/// <para><b>Key insight:</b> For a stable distribution, p + s should remain relatively constant.
/// A significant increase indicates the underlying distribution has changed.</para>
///
/// <para><b>Parameters:</b>
/// <list type="bullet">
/// <item><b>warningThreshold:</b> Number of standard deviations for warning (default: 2)</item>
/// <item><b>driftThreshold:</b> Number of standard deviations for drift (default: 3)</item>
/// </list>
/// </para>
///
/// <para><b>Advantages:</b>
/// <list type="bullet">
/// <item>Simple and computationally efficient</item>
/// <item>Two-stage detection (warning before drift)</item>
/// <item>Well-suited for sudden drift</item>
/// </list>
/// </para>
///
/// <para><b>Limitations:</b>
/// <list type="bullet">
/// <item>Assumes binary errors (0 or 1)</item>
/// <item>May be slow to detect gradual drift</item>
/// <item>Can miss drift if error rate decreases</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Gama et al. (2004). "Learning with Drift Detection"</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DDMDriftDetector<T> : DriftDetectorBase<T>
{
    private readonly double _warningThreshold;
    private readonly double _driftThreshold;

    /// <summary>
    /// Running sum of errors.
    /// </summary>
    private double _errorSum;

    /// <summary>
    /// Minimum value of p + s observed.
    /// </summary>
    private double _minPPlusS;

    /// <summary>
    /// Error rate p at minimum p + s.
    /// </summary>
    private double _pAtMin;

    /// <summary>
    /// Standard deviation s at minimum p + s.
    /// </summary>
    private double _sAtMin;

    /// <summary>
    /// Sample count at the warning point.
    /// </summary>
    private int _warningStartCount;

    /// <summary>
    /// Gets whether the detector is in the warning zone.
    /// </summary>
    public new bool IsInWarning { get; private set; }

    /// <summary>
    /// Creates a new DDM drift detector.
    /// </summary>
    /// <param name="warningThreshold">Number of standard deviations for warning (default: 2).</param>
    /// <param name="driftThreshold">Number of standard deviations for drift (default: 3).</param>
    /// <param name="minimumObservations">Minimum samples before detection starts (default: 30).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Higher thresholds make the detector less sensitive (fewer false alarms
    /// but may miss subtle drift). Lower thresholds make it more sensitive (more false alarms but
    /// catches drift faster).</para>
    /// </remarks>
    public DDMDriftDetector(double warningThreshold = 2.0, double driftThreshold = 3.0, int minimumObservations = 30)
    {
        _warningThreshold = warningThreshold;
        _driftThreshold = driftThreshold;
        MinimumObservations = minimumObservations;
        Reset();
    }

    /// <summary>
    /// Adds a new observation (typically a prediction error: 1 for wrong, 0 for correct).
    /// </summary>
    /// <param name="value">Error indicator (0 = correct, 1 = error).</param>
    /// <returns>True if drift is detected.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Feed this method with your classifier's errors:
    /// <code>
    /// var detector = new DDMDriftDetector&lt;double&gt;();
    /// foreach (var (x, y) in testData)
    /// {
    ///     var prediction = model.Predict(x);
    ///     double error = prediction != y ? 1 : 0;
    ///     if (detector.AddObservation(error))
    ///     {
    ///         // Drift detected - retrain model
    ///     }
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public override bool AddObservation(T value)
    {
        double x = ToDouble(value);
        ObservationCount++;
        _errorSum += x;

        // Calculate error rate and standard deviation
        double p = _errorSum / ObservationCount;
        double s = Math.Sqrt(p * (1 - p) / ObservationCount);
        double pPlusS = p + s;

        EstimatedMean = p;

        // Only check for drift after minimum observations
        if (ObservationCount < MinimumObservations)
        {
            _minPPlusS = pPlusS;
            _pAtMin = p;
            _sAtMin = s;
            return false;
        }

        // Track the minimum p + s
        if (pPlusS < _minPPlusS)
        {
            _minPPlusS = pPlusS;
            _pAtMin = p;
            _sAtMin = s;
            IsInWarning = false;
            _warningStartCount = 0;
        }

        // Calculate thresholds
        double warningLevel = _pAtMin + _sAtMin + _warningThreshold * _sAtMin;
        double driftLevel = _pAtMin + _sAtMin + _driftThreshold * _sAtMin;

        // Calculate drift probability as normalized distance
        if (driftLevel > _minPPlusS)
        {
            DriftProbability = Math.Min(1.0, Math.Max(0.0, (pPlusS - _minPPlusS) / (driftLevel - _minPPlusS)));
        }

        // Check for drift
        if (pPlusS > driftLevel)
        {
            IsInDrift = true;
            IsInWarning = false;
            return true;
        }

        // Check for warning
        if (pPlusS > warningLevel)
        {
            if (!IsInWarning)
            {
                IsInWarning = true;
                _warningStartCount = ObservationCount;
            }
        }
        else
        {
            IsInWarning = false;
            _warningStartCount = 0;
        }

        return false;
    }

    /// <summary>
    /// Resets the detector to its initial state.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        _errorSum = 0;
        _minPPlusS = double.MaxValue;
        _pAtMin = 0;
        _sAtMin = 0;
        IsInWarning = false;
        _warningStartCount = 0;
    }

    /// <summary>
    /// Gets the current error rate.
    /// </summary>
    public double ErrorRate => ObservationCount > 0 ? _errorSum / ObservationCount : 0;

    /// <summary>
    /// Gets the minimum p + s observed.
    /// </summary>
    public double MinimumPPlusS => _minPPlusS;

    /// <summary>
    /// Gets the number of samples since entering warning zone.
    /// </summary>
    public int SamplesInWarning => IsInWarning ? ObservationCount - _warningStartCount : 0;
}
