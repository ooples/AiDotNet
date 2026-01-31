namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Value-at-Risk (VaR) models.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> VaR options define how conservative your risk estimate is
/// and how much data the model expects. Think of them as the "risk dial" and the
/// "input size" settings.
/// </para>
/// </remarks>
public class VaROptions<T>
{
    /// <summary>
    /// The confidence level for VaR calculation (e.g., 0.95 or 0.99).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> 0.95 means "I want to be safe 95% of the time."
    /// Higher values mean a more conservative (larger) loss estimate.
    /// </para>
    /// </remarks>
    public double ConfidenceLevel { get; set; } = 0.95;

    /// <summary>
    /// The time horizon for the risk assessment (in days).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how far into the future the risk is measured.
    /// 1 = tomorrow, 10 = the next 10 days.
    /// </para>
    /// </remarks>
    public int TimeHorizon { get; set; } = 1;

    /// <summary>
    /// Number of input features used for risk calculation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many columns your input data has.
    /// For example, if you feed 10 indicators, set this to 10.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 10;

    /// <summary>
    /// Validates the VaR options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method checks that your settings make sense
    /// (for example, confidence must be between 0 and 1).
    /// It prevents confusing runtime errors later.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        if (ConfidenceLevel <= 0 || ConfidenceLevel >= 1)
            throw new ArgumentException("ConfidenceLevel must be between 0 and 1.", nameof(ConfidenceLevel));
        if (TimeHorizon < 1)
            throw new ArgumentException("TimeHorizon must be at least 1.", nameof(TimeHorizon));
        if (NumFeatures < 1)
            throw new ArgumentException("NumFeatures must be at least 1.", nameof(NumFeatures));
    }
}
