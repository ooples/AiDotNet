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
    private readonly int _minimumErrors;

    private int _lastErrorPosition;
    private int _errorCount;
    private double _distanceMean;
    private double _distanceM2;  // For Welford's algorithm
    private double _maxDistancePsi;
    private double _maxDistanceMean;
    private double _maxDistanceStd;
    private int _sinceLastError;

    /// <summary>
    /// Creates a new EDDM drift detector.
    /// </summary>
    /// <param name="warningThreshold">Ratio threshold for warning (default: 0.95).</param>
    /// <param name="driftThreshold">Ratio threshold for drift (default: 0.90).</param>
    /// <param name="minimumObservations">Minimum samples before detection starts (default: 30).</param>
    /// <param name="minimumErrors">Minimum errors needed before detection starts (default: 30).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> These thresholds are ratios (0-1). When the current metric
    /// falls below these fractions of the best observed metric, warnings/drift are triggered.
    /// Lower thresholds mean more tolerance for degradation.</para>
    /// </remarks>
    public EDDMDriftDetector(double warningThreshold = 0.95, double driftThreshold = 0.90, int minimumObservations = 30, int minimumErrors = 30)
    {
        if (warningThreshold <= 0 || warningThreshold >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(warningThreshold),
                "Warning threshold must be in range (0, 1).");
        }

        if (driftThreshold <= 0 || driftThreshold >= warningThreshold)
        {
            throw new ArgumentOutOfRangeException(nameof(driftThreshold),
                "Drift threshold must be in range (0, warningThreshold).");
        }

        if (minimumErrors < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(minimumErrors),
                "Minimum errors must be at least 2 (to compute distances).");
        }

        _warningThreshold = warningThreshold;
        _driftThreshold = driftThreshold;
        _minimumErrors = minimumErrors;
        MinimumObservations = minimumObservations;
        Reset();
    }

    /// <summary>
    /// Gets the minimum number of errors required before detection starts.
    /// </summary>
    public int MinimumErrors => _minimumErrors;

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
        double val = ToDouble(value);
        ObservationCount++;
        _sinceLastError++;

        bool isError = val > 0.5;

        if (isError)
        {
            _errorCount++;

            if (_lastErrorPosition > 0)
            {
                // Calculate distance since last error
                int distance = ObservationCount - _lastErrorPosition;

                // Update running statistics using Welford's algorithm
                double delta = distance - _distanceMean;
                _distanceMean += delta / (_errorCount - 1);
                double delta2 = distance - _distanceMean;
                _distanceM2 += delta * delta2;
            }

            _lastErrorPosition = ObservationCount;
            _sinceLastError = 0;
        }

        // Need at least minimum errors to compute distance statistics
        if (_errorCount < _minimumErrors || ObservationCount < MinimumObservations)
        {
            return false;
        }

        // Calculate current statistics
        double variance = _errorCount > 2 ? _distanceM2 / (_errorCount - 2) : 0;
        double std = Math.Sqrt(Math.Max(0, variance));
        double psi = _distanceMean + 2 * std;

        EstimatedMean = 1.0 / Math.Max(1, _distanceMean);  // Convert to error rate approximation

        // Update maximum if new best
        if (psi > _maxDistancePsi)
        {
            _maxDistancePsi = psi;
            _maxDistanceMean = _distanceMean;
            _maxDistanceStd = std;
        }

        // Calculate ratio to maximum
        double ratio = _maxDistancePsi > 0 ? psi / _maxDistancePsi : 1.0;

        // Check for drift (ratio dropping means distances are shrinking = more errors)
        if (ratio < _driftThreshold)
        {
            IsInDrift = true;
            IsInWarning = false;
            DriftProbability = 1.0;
            return true;
        }

        // Check for warning
        if (ratio < _warningThreshold)
        {
            IsInWarning = true;
            // Estimate drift probability
            double range = _warningThreshold - _driftThreshold;
            DriftProbability = range > 0 ? (_warningThreshold - ratio) / range : 0.5;
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
    /// Resets the detector to its initial state.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        _lastErrorPosition = 0;
        _errorCount = 0;
        _distanceMean = 0;
        _distanceM2 = 0;
        _maxDistancePsi = 0;
        _maxDistanceMean = 0;
        _maxDistanceStd = 0;
        _sinceLastError = 0;
    }

    /// <summary>
    /// Gets the number of errors detected.
    /// </summary>
    public int ErrorCount => _errorCount;

    /// <summary>
    /// Gets the average distance between errors.
    /// </summary>
    public double AverageErrorDistance => _errorCount > 1 ? _distanceMean : double.PositiveInfinity;

    /// <summary>
    /// Gets the maximum p' + 2s' observed.
    /// </summary>
    public double MaximumPPrime2S => _maxDistancePsi;

    /// <summary>
    /// Gets the current ratio (lower means closer to drift).
    /// </summary>
    public double CurrentRatio
    {
        get
        {
            if (_errorCount < 2 || _maxDistancePsi <= 0) return 1.0;

            double variance = _errorCount > 2 ? _distanceM2 / (_errorCount - 2) : 0;
            double std = Math.Sqrt(Math.Max(0, variance));
            double psi = _distanceMean + 2 * std;

            return psi / _maxDistancePsi;
        }
    }
}
