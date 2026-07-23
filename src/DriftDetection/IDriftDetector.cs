namespace AiDotNet.DriftDetection;

/// <summary>
/// Interface for concept drift detectors in streaming/online learning scenarios.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Concept drift occurs when the statistical properties of data
/// change over time. For example, customer behavior patterns may shift seasonally, or
/// spam email characteristics may evolve. Drift detectors monitor incoming data and alert
/// when significant changes are detected, signaling that a model may need retraining.</para>
///
/// <para><b>Types of Drift:</b>
/// <list type="bullet">
/// <item><b>Sudden drift:</b> Abrupt change (e.g., system upgrade, policy change)</item>
/// <item><b>Gradual drift:</b> Slow transition between concepts over time</item>
/// <item><b>Incremental drift:</b> Small continuous changes that accumulate</item>
/// <item><b>Recurring drift:</b> Concepts that reappear periodically (e.g., seasonal)</item>
/// </list>
/// </para>
///
/// <para><b>Common Approaches:</b>
/// <list type="bullet">
/// <item><b>Error-rate based:</b> Monitor classifier errors (DDM, EDDM)</item>
/// <item><b>Distribution-based:</b> Compare data distributions over windows (ADWIN)</item>
/// <item><b>Sequential analysis:</b> Cumulative sum tests (Page-Hinkley)</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b> Any online learning scenario where data distribution may change:
/// fraud detection, recommendation systems, sensor monitoring, financial trading, etc.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("DriftDetectorStandalone")]
public interface IDriftDetector<T>
{
    /// <summary>
    /// Adds a new observation to the drift detector.
    /// </summary>
    /// <param name="value">The new value to process (often an error indicator or prediction).</param>
    /// <returns>True if drift was detected with this observation, false otherwise.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this method for each new data point (often the error
    /// from your model's prediction). The detector will update its internal state and return
    /// true if it detects that drift has occurred.</para>
    /// </remarks>
    bool AddObservation(T value);

    /// <summary>
    /// Gets whether drift has been detected.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This indicates the current drift state. Once drift is detected,
    /// this remains true until <see cref="Reset"/> is called.</para>
    /// </remarks>
    bool IsInDrift { get; }

    /// <summary>
    /// Gets whether a warning signal has been triggered (pre-drift indicator).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some detectors (like DDM) have a warning zone before full drift
    /// detection. When IsInWarning is true, drift may be imminent. Use this for proactive measures.</para>
    /// </remarks>
    bool IsInWarning { get; }

    /// <summary>
    /// Gets the estimated probability that drift has occurred.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This provides a continuous confidence measure rather than
    /// just a binary drift/no-drift decision. Useful for monitoring trends.</para>
    /// </remarks>
    double DriftProbability { get; }

    /// <summary>
    /// Resets the detector to its initial state.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this after handling a detected drift (e.g., after
    /// retraining your model). This clears all history and allows the detector to start fresh.</para>
    /// </remarks>
    void Reset();

    /// <summary>
    /// Gets the current estimated mean of the stream.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the running estimate of the average value in the
    /// stream. Useful for understanding the current data distribution.</para>
    /// </remarks>
    double EstimatedMean { get; }

    /// <summary>
    /// Gets the total number of observations processed since the last reset.
    /// </summary>
    int ObservationCount { get; }
}
