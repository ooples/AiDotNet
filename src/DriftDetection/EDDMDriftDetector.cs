namespace AiDotNet.DriftDetection;

/// <summary>
/// Early Drift Detection Method (EDDM) for concept drift detection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> EDDM is an improvement over DDM that monitors the distance between
/// errors rather than just the error rate. This makes it better at detecting gradual drift because
/// even if the error rate stays similar, the spacing between errors may change.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Track the distance (number of samples) between consecutive errors</item>
/// <item>Calculate the mean distance p' and standard deviation s'</item>
/// <item>Monitor p' + 2s' and remember its maximum value</item>
/// <item>Warning: when (p' + 2s') / (p'_max + 2s'_max) drops below warning threshold</item>
/// <item>Drift: when (p' + 2s') / (p'_max + 2s'_max) drops below drift threshold</item>
/// </list>
/// </para>
///
/// <para><b>Key insight:</b> When performance is good, errors are far apart (high mean distance).
/// When drift occurs, errors become more frequent and closer together (lower mean distance).</para>
///
/// <para><b>Advantages over DDM:</b>
/// <list type="bullet">
/// <item>Better at detecting gradual drift</item>
/// <item>More sensitive to changes in error patterns</item>
/// <item>Can detect drift even when error rate is low</item>
/// </list>
/// </para>
///
/// <para><b>Parameters:</b>
/// <list type="bullet">
/// <item><b>warningThreshold:</b> Ratio for warning level (default: 0.95)</item>
/// <item><b>driftThreshold:</b> Ratio for drift level (default: 0.90)</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Baena-García et al. (2006). "Early Drift Detection Method"</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class EDDMDriftDetector<T> : DriftDetectorBase<T>
{
    private readonly double _warningThreshold;
    private readonly double _driftThreshold;

    /// <summary>
    /// Count of samples since the last error.
    /// </summary>
    private int _distanceSinceLastError;

    /// <summary>
    /// Number of errors observed.
    /// </summary>
    private int _errorCount;

    /// <summary>
    /// Running sum of distances between errors.
    /// </summary>
    private double _distanceSum;

    /// <summary>
    /// Running sum of squared distances (for variance calculation).
    /// </summary>
    private double _distanceSumSquared;

    /// <summary>
    /// Maximum value of p' + 2s' observed.
    /// </summary>
    private double _maxPPrime2S;

    /// <summary>
    /// Mean distance at maximum.
    /// </summary>
    private double _pPrimeAtMax;

    /// <summary>
    /// Standard deviation at maximum.
    /// </summary>
    private double _sPrimeAtMax;

    /// <summary>
    /// Observation count when warning was first triggered.
    /// </summary>
    private int _warningStartCount;

    /// <summary>
    /// Gets whether the detector is in the warning zone.
    /// </summary>
    public new bool IsInWarning { get; private set; }

    /// <summary>
    /// Creates a new EDDM drift detector.
    /// </summary>
    /// <param name="warningThreshold">Ratio threshold for warning (default: 0.95).</param>
    /// <param name="driftThreshold">Ratio threshold for drift (default: 0.90).</param>
    /// <param name="minimumObservations">Minimum samples before detection starts (default: 30).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> These thresholds are ratios (0-1). When the current metric
    /// falls below these fractions of the best observed metric, warnings/drift are triggered.
    /// Lower thresholds mean more tolerance for degradation.</para>
    /// </remarks>
    public EDDMDriftDetector(double warningThreshold = 0.95, double driftThreshold = 0.90, int minimumObservations = 30)
    {
        if (warningThreshold <= driftThreshold)
        {
            throw new ArgumentException("Warning threshold must be greater than drift threshold.");
        }

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
    /// <para><b>For Beginners:</b> EDDM tracks the distance between consecutive errors.
    /// If errors suddenly start occurring closer together, it indicates potential drift.</para>
    /// </remarks>
    public override bool AddObservation(T value)
    {
        double x = ToDouble(value);
        ObservationCount++;
        _distanceSinceLastError++;

        // Check if this is an error (value > 0.5 for binary, or > threshold)
        bool isError = x > 0.5;

        if (!isError)
        {
            // Not an error, just update observation count
            return false;
        }

        // This is an error - record the distance
        _errorCount++;

        // Need at least 2 errors to measure distance
        if (_errorCount < 2)
        {
            _distanceSinceLastError = 0;
            return false;
        }

        // Update running statistics for distance
        double distance = _distanceSinceLastError;
        _distanceSum += distance;
        _distanceSumSquared += distance * distance;

        // Reset distance counter
        _distanceSinceLastError = 0;

        // Calculate mean and standard deviation of distances
        int distanceCount = _errorCount - 1; // Number of distance measurements
        double pPrime = _distanceSum / distanceCount;
        double variance = (_distanceSumSquared / distanceCount) - (pPrime * pPrime);
        double sPrime = Math.Sqrt(Math.Max(0, variance));

        EstimatedMean = 1.0 / pPrime; // Convert to error rate

        // Calculate p' + 2s'
        double pPrime2S = pPrime + 2 * sPrime;

        // Only check for drift after minimum observations
        if (_errorCount < MinimumObservations)
        {
            if (pPrime2S > _maxPPrime2S)
            {
                _maxPPrime2S = pPrime2S;
                _pPrimeAtMax = pPrime;
                _sPrimeAtMax = sPrime;
            }
            return false;
        }

        // Update maximum if we found a better state
        if (pPrime2S > _maxPPrime2S)
        {
            _maxPPrime2S = pPrime2S;
            _pPrimeAtMax = pPrime;
            _sPrimeAtMax = sPrime;
            IsInWarning = false;
            _warningStartCount = 0;
        }

        // Calculate ratio (current / maximum)
        double ratio = _maxPPrime2S > 0 ? pPrime2S / _maxPPrime2S : 1.0;

        // Update drift probability
        DriftProbability = Math.Min(1.0, Math.Max(0.0, 1.0 - (ratio - _driftThreshold) / (_warningThreshold - _driftThreshold)));

        // Check for drift
        if (ratio < _driftThreshold)
        {
            IsInDrift = true;
            IsInWarning = false;
            return true;
        }

        // Check for warning
        if (ratio < _warningThreshold)
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
        _distanceSinceLastError = 0;
        _errorCount = 0;
        _distanceSum = 0;
        _distanceSumSquared = 0;
        _maxPPrime2S = 0;
        _pPrimeAtMax = 0;
        _sPrimeAtMax = 0;
        IsInWarning = false;
        _warningStartCount = 0;
    }

    /// <summary>
    /// Gets the number of errors detected.
    /// </summary>
    public int ErrorCount => _errorCount;

    /// <summary>
    /// Gets the average distance between errors.
    /// </summary>
    public double AverageErrorDistance => _errorCount > 1 ? _distanceSum / (_errorCount - 1) : double.PositiveInfinity;

    /// <summary>
    /// Gets the maximum p' + 2s' observed.
    /// </summary>
    public double MaximumPPrime2S => _maxPPrime2S;

    /// <summary>
    /// Gets the current ratio (lower means closer to drift).
    /// </summary>
    public double CurrentRatio
    {
        get
        {
            if (_errorCount < 2 || _maxPPrime2S <= 0) return 1.0;

            int distanceCount = _errorCount - 1;
            double pPrime = _distanceSum / distanceCount;
            double variance = (_distanceSumSquared / distanceCount) - (pPrime * pPrime);
            double sPrime = Math.Sqrt(Math.Max(0, variance));
            double pPrime2S = pPrime + 2 * sPrime;

            return pPrime2S / _maxPPrime2S;
        }
    }

    /// <summary>
    /// Gets the number of samples since entering warning zone.
    /// </summary>
    public int SamplesInWarning => IsInWarning ? ObservationCount - _warningStartCount : 0;
}
