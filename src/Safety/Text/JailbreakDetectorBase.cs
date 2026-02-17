namespace AiDotNet.Safety.Text;

/// <summary>
/// Abstract base class for jailbreak and prompt injection detection modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for jailbreak detectors including sensitivity
/// configuration and common scoring utilities. Concrete implementations provide
/// the actual detection algorithm (pattern, semantic, gradient, ensemble).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all jailbreak detectors.
/// Each detector type extends this and adds its own way of catching people trying to
/// trick the AI into ignoring its safety rules.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class JailbreakDetectorBase<T> : TextSafetyModuleBase<T>, IJailbreakDetector<T>
{
    /// <summary>
    /// The detection sensitivity level (0.0 = lenient, 1.0 = strict).
    /// </summary>
    protected readonly double Sensitivity;

    /// <summary>
    /// Initializes the jailbreak detector base with a sensitivity level.
    /// </summary>
    /// <param name="sensitivity">Detection sensitivity (0.0 to 1.0). Default: 0.5.</param>
    protected JailbreakDetectorBase(double sensitivity = 0.5)
    {
        Sensitivity = sensitivity;
    }

    /// <inheritdoc />
    public abstract double GetJailbreakScore(string text);
}
