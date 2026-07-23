namespace AiDotNet.DriftDetection;

/// <summary>
/// Implements EDDM (Early Drift Detection Method) for concept drift detection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> EDDM is an improvement over DDM that focuses on detecting
/// drift earlier, especially for gradual drift. Instead of monitoring just the error rate,
/// EDDM monitors the distance (number of samples) between consecutive errors.</para>
///
/// <para><b>How EDDM works:</b>
/// <list type="number">
/// <item>Track the distance between consecutive errors (not just error rate)</item>
/// <item>Maintain running mean (p') and standard deviation (s') of these distances</item>
/// <item>Monitor the ratio: current (p' + 2*s') / maximum (p' + 2*s')</item>
/// <item>Warning: ratio drops below warning threshold (e.g., 0.95)</item>
/// <item>Drift: ratio drops below drift threshold (e.g., 0.90)</item>
/// </list>
/// </para>
///
/// <para><b>Why distance between errors?</b> When a model starts degrading:
/// <list type="bullet">
/// <item>Errors become more frequent → distances decrease</item>
/// <item>Error pattern becomes more consistent → standard deviation decreases</item>
/// <item>Both effects contribute to earlier detection than pure error rate</item>
/// </list>
/// </para>
///
/// <para><b>Advantages over DDM:</b>
/// <list type="bullet">
/// <item>Earlier detection of gradual drift</item>
/// <item>More stable in presence of noise</item>
/// <item>Works well when errors are relatively rare</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Baena-García et al., "Early Drift Detection Method" (2006)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class EDDM<T> : DriftDetectorBase<T>
{
    private int _lastErrorPosition;
    private int _errorCount;
    private double _distanceMean;
    private double _distanceM2;  // For Welford's algorithm
    private double _maxDistancePsi;
    private double _maxDistanceMean;
    private double _maxDistanceStd;
    private int _sinceLastError;

    /// <summary>
    /// Gets the warning threshold (ratio of current to maximum p' + 2s').
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the ratio drops below this value, a warning is
    /// issued. Typical value: 0.95 (95% of best performance).</para>
    /// </remarks>
    public double WarningThreshold { get; }

    /// <summary>
    /// Gets the drift threshold (ratio of current to maximum p' + 2s').
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the ratio drops below this value, drift is
    /// declared. Typical value: 0.90 (90% of best performance).</para>
    /// </remarks>
    public double DriftThreshold { get; }

    /// <summary>
    /// Creates a new EDDM drift detector.
    /// </summary>
    /// <param name="warningThreshold">Ratio threshold for warning (default: 0.95).</param>
    /// <param name="driftThreshold">Ratio threshold for drift (default: 0.90).</param>
    /// <param name="minimumObservations">Minimum samples before detection starts (default: 30).</param>
    /// <param name="minimumErrors">Minimum errors needed before detection starts (default: 30).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The default parameters work well for most scenarios.
    /// Increase thresholds (closer to 1.0) for faster but less reliable detection.
    /// Decrease for more conservative detection.</para>
    /// </remarks>
    public EDDM(
        double warningThreshold = 0.95,
        double driftThreshold = 0.90,
        int minimumObservations = 30,
        int minimumErrors = 30)
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

        if (minimumObservations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(minimumObservations),
                "Minimum observations must be at least 1.");
        }

        if (minimumErrors < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(minimumErrors),
                "Minimum errors must be at least 2 (to compute distances).");
        }

        WarningThreshold = warningThreshold;
        DriftThreshold = driftThreshold;
        MinimumObservations = minimumObservations;
        MinimumErrors = minimumErrors;

        Reset();
    }

    /// <summary>
    /// Gets the minimum number of errors required before detection starts.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> EDDM needs enough errors to compute meaningful statistics
    /// on the distances between them.</para>
    /// </remarks>
    public int MinimumErrors { get; }

    /// <inheritdoc />
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
    /// Adds a new observation (error indicator) to the detector.
    /// </summary>
    /// <param name="value">The observation: typically 1.0 for error, 0.0 for correct.</param>
    /// <returns>True if drift was detected.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Pass 1 when your classifier makes an error, 0 when correct.
    /// EDDM tracks the spacing between errors to detect drift earlier than error rate alone.</para>
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

        // Need at least 2 errors to compute distance statistics
        if (_errorCount < MinimumErrors || ObservationCount < MinimumObservations)
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
        if (ratio < DriftThreshold)
        {
            IsInDrift = true;
            IsInWarning = false;
            DriftProbability = 1.0;
            return true;
        }

        // Check for warning
        if (ratio < WarningThreshold)
        {
            IsInWarning = true;
            // Estimate drift probability
            double range = WarningThreshold - DriftThreshold;
            DriftProbability = range > 0 ? (WarningThreshold - ratio) / range : 0.5;
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
    /// Gets the current mean distance between errors.
    /// </summary>
    /// <returns>Average number of samples between consecutive errors.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Higher values mean errors are rare (good).
    /// Lower values mean errors are frequent (potential drift).</para>
    /// </remarks>
    public double GetMeanDistance()
    {
        return _distanceMean;
    }

    /// <summary>
    /// Gets the current standard deviation of distances between errors.
    /// </summary>
    /// <returns>Standard deviation of inter-error distances.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lower values mean errors occur at regular intervals.
    /// Higher values mean error timing is unpredictable.</para>
    /// </remarks>
    public double GetDistanceStd()
    {
        if (_errorCount <= 2) return 0;
        double variance = _distanceM2 / (_errorCount - 2);
        return Math.Sqrt(Math.Max(0, variance));
    }

    /// <summary>
    /// Gets the total number of errors observed.
    /// </summary>
    public int ErrorCount => _errorCount;

    /// <summary>
    /// Gets the current ratio of performance to maximum observed.
    /// </summary>
    /// <returns>Ratio in range (0, 1] where 1 means at best performance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Track this value over time. When it drops significantly,
    /// drift is occurring. Values close to 1.0 indicate stable performance.</para>
    /// </remarks>
    public double GetCurrentRatio()
    {
        if (_errorCount < 2 || _maxDistancePsi <= 0) return 1.0;

        double variance = _errorCount > 2 ? _distanceM2 / (_errorCount - 2) : 0;
        double std = Math.Sqrt(Math.Max(0, variance));
        double psi = _distanceMean + 2 * std;

        return psi / _maxDistancePsi;
    }
}
