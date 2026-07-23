using AiDotNet.Helpers;

namespace AiDotNet.DriftDetection;

/// <summary>
/// Base class for drift detectors providing common functionality.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides the common infrastructure needed by
/// all drift detectors: tracking observations, managing state, and calculating statistics.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class DriftDetectorBase<T> : IDriftDetector<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This allows the detector to work with any numeric type
    /// (float, double, decimal) by providing generic math operations.</para>
    /// </remarks>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets or sets whether drift has been detected.
    /// </summary>
    public bool IsInDrift { get; protected set; }

    /// <summary>
    /// Gets or sets whether a warning has been triggered.
    /// </summary>
    public bool IsInWarning { get; protected set; }

    /// <summary>
    /// Gets or sets the estimated drift probability.
    /// </summary>
    public double DriftProbability { get; protected set; }

    /// <summary>
    /// Gets or sets the estimated mean of observations.
    /// </summary>
    public double EstimatedMean { get; protected set; }

    /// <summary>
    /// Gets the total observation count.
    /// </summary>
    public int ObservationCount { get; protected set; }

    /// <summary>
    /// Minimum number of observations required before drift detection begins.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Drift detection requires enough samples to establish
    /// a baseline. This threshold prevents false alarms during the initial "warm-up" period.</para>
    /// </remarks>
    protected int MinimumObservations { get; set; } = 30;

    /// <summary>
    /// Creates a new drift detector base.
    /// </summary>
    protected DriftDetectorBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Adds a new observation to the detector.
    /// </summary>
    /// <param name="value">The new observation value.</param>
    /// <returns>True if drift was detected.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method updates the detector's internal state and
    /// checks for drift. The specific algorithm is implemented in derived classes.</para>
    /// </remarks>
    public abstract bool AddObservation(T value);

    /// <summary>
    /// Resets the detector to its initial state.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this after drift is detected and handled. This
    /// clears all history so the detector can start fresh.</para>
    /// </remarks>
    public virtual void Reset()
    {
        IsInDrift = false;
        IsInWarning = false;
        DriftProbability = 0;
        EstimatedMean = 0;
        ObservationCount = 0;
    }

    /// <summary>
    /// Converts a value to double for calculations.
    /// </summary>
    /// <param name="value">The value to convert.</param>
    /// <returns>The double representation.</returns>
    protected double ToDouble(T value)
    {
        return NumOps.ToDouble(value);
    }
}
