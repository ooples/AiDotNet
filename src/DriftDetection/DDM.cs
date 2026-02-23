namespace AiDotNet.DriftDetection;

/// <summary>
/// Implements DDM (Drift Detection Method) for concept drift detection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> DDM is one of the simplest and most popular drift detectors.
/// It monitors the error rate of a classifier and triggers when the error rate increases
/// significantly compared to the minimum observed error rate.</para>
///
/// <para><b>How DDM works:</b>
/// <list type="number">
/// <item>Track the error rate (p) and its standard deviation (s)</item>
/// <item>Remember the minimum p + s observed (the "best" state)</item>
/// <item>Warning: Current p + s exceeds minimum by more than 2σ</item>
/// <item>Drift: Current p + s exceeds minimum by more than 3σ</item>
/// </list>
/// </para>
///
/// <para><b>Key Concepts:</b>
/// <list type="bullet">
/// <item><b>Error rate (p):</b> Running average of errors (0 = correct, 1 = error)</item>
/// <item><b>Standard deviation (s):</b> sqrt(p * (1-p) / n)</item>
/// <item><b>Minimum p + s:</b> The best observed performance</item>
/// <item><b>Warning zone:</b> Performance degraded but not enough for drift</item>
/// <item><b>Drift zone:</b> Significant performance degradation detected</item>
/// </list>
/// </para>
///
/// <para><b>Advantages:</b> Simple, fast, low memory, works well with gradual drift.</para>
/// <para><b>Disadvantages:</b> Only works with error rates (0/1), may miss small drifts.</para>
///
/// <para><b>Reference:</b> Gama et al., "Learning with Drift Detection" (2004)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DDM<T> : DriftDetectorBase<T>
{
    private int _numErrors;
    private double _minPSi;
    private double _minP;
    private double _minS;
    private int _minCount;

    /// <summary>
    /// Gets the warning threshold (default: 2.0 standard deviations).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the current p + s exceeds the minimum by this many
    /// standard deviations, a warning is issued. Lower values = more sensitive.</para>
    /// </remarks>
    public double WarningThreshold { get; }

    /// <summary>
    /// Gets the drift threshold (default: 3.0 standard deviations).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the current p + s exceeds the minimum by this many
    /// standard deviations, drift is declared. Lower values = more sensitive.</para>
    /// </remarks>
    public double DriftThreshold { get; }

    /// <summary>
    /// Gets the delay between warning and drift detection (for gradual drift).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If drift is not confirmed within this many observations
    /// after a warning, the warning is cleared. This prevents false alarms from noise.</para>
    /// </remarks>
    public int WarningDelay { get; }

    private int _warningStartCount;

    /// <summary>
    /// Creates a new DDM drift detector.
    /// </summary>
    /// <param name="warningThreshold">Standard deviations for warning (default: 2.0).</param>
    /// <param name="driftThreshold">Standard deviations for drift (default: 3.0).</param>
    /// <param name="minimumObservations">Minimum samples before detection starts (default: 30).</param>
    /// <param name="warningDelay">Samples before warning expires (default: 100).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The default parameters work well for most scenarios.
    /// Decrease thresholds for faster detection (but more false alarms).
    /// Increase warningDelay if drift happens slowly.</para>
    /// </remarks>
    public DDM(
        double warningThreshold = 2.0,
        double driftThreshold = 3.0,
        int minimumObservations = 30,
        int warningDelay = 100)
    {
        if (warningThreshold <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(warningThreshold), "Warning threshold must be positive.");
        }

        if (driftThreshold <= warningThreshold)
        {
            throw new ArgumentOutOfRangeException(nameof(driftThreshold),
                "Drift threshold must be greater than warning threshold.");
        }

        if (minimumObservations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(minimumObservations),
                "Minimum observations must be at least 1.");
        }

        WarningThreshold = warningThreshold;
        DriftThreshold = driftThreshold;
        MinimumObservations = minimumObservations;
        WarningDelay = warningDelay;

        Reset();
    }

    /// <inheritdoc />
    public override void Reset()
    {
        base.Reset();
        _numErrors = 0;
        _minPSi = double.MaxValue;
        _minP = 0;
        _minS = 0;
        _minCount = 0;
        _warningStartCount = 0;
    }

    /// <summary>
    /// Adds a new observation (error indicator) to the detector.
    /// </summary>
    /// <param name="value">The observation: typically 1.0 for error, 0.0 for correct.</param>
    /// <returns>True if drift was detected.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Pass 1 (or any positive number) when your classifier
    /// makes an error, and 0 when it predicts correctly. DDM will track the error rate
    /// and detect when it increases significantly.</para>
    /// </remarks>
    public override bool AddObservation(T value)
    {
        double val = ToDouble(value);
        ObservationCount++;

        // Count as error if value > 0.5 (allows binary or probabilistic input)
        if (val > 0.5)
        {
            _numErrors++;
        }

        // Calculate current error rate and standard deviation
        double p = (double)_numErrors / ObservationCount;
        double s = Math.Sqrt(p * (1 - p) / ObservationCount);
        double psi = p + s;

        EstimatedMean = p;

        // Wait for minimum observations
        if (ObservationCount < MinimumObservations)
        {
            return false;
        }

        // Update minimum if we have a new best
        if (psi < _minPSi)
        {
            _minPSi = psi;
            _minP = p;
            _minS = s;
            _minCount = ObservationCount;
        }

        // Calculate threshold for drift detection
        double driftBound = _minP + DriftThreshold * _minS;
        double warningBound = _minP + WarningThreshold * _minS;

        // Check for drift
        if (p > driftBound)
        {
            IsInDrift = true;
            IsInWarning = false;
            DriftProbability = 1.0;
            return true;
        }

        // Check for warning
        if (p > warningBound)
        {
            if (!IsInWarning)
            {
                IsInWarning = true;
                _warningStartCount = ObservationCount;
            }
            else if (ObservationCount - _warningStartCount > WarningDelay)
            {
                // Warning expired without drift - could be noise
                // Reset warning but keep monitoring
                IsInWarning = false;
            }

            // Estimate drift probability based on how close we are to drift threshold
            double range = driftBound - warningBound;
            if (range > 0)
            {
                DriftProbability = (p - warningBound) / range;
            }
            else
            {
                DriftProbability = 0.5;
            }
        }
        else
        {
            IsInWarning = false;
            DriftProbability = 0;
        }

        IsInDrift = false;
        return false;
    }

    /// <summary>
    /// Gets the current error rate.
    /// </summary>
    /// <returns>The proportion of errors observed.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the running average of errors. A value of 0.1
    /// means 10% of predictions were wrong.</para>
    /// </remarks>
    public double GetErrorRate()
    {
        return ObservationCount > 0 ? (double)_numErrors / ObservationCount : 0;
    }

    /// <summary>
    /// Gets the minimum error rate observed (the baseline).
    /// </summary>
    /// <returns>The minimum error rate + standard deviation observed.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the "best" performance the model achieved.
    /// Drift is detected when current performance is significantly worse than this.</para>
    /// </remarks>
    public double GetMinimumPsi()
    {
        return _minPSi < double.MaxValue ? _minPSi : 0;
    }
}
