using AiDotNet.Safety.Text;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Abstract base class for toxicity detection modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for toxicity detectors including the toxicity
/// threshold configuration and common scoring utilities. Concrete implementations
/// provide the actual detection algorithm (rule-based, embedding, classifier, ensemble).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides shared code for all toxicity detectors.
/// Each detector type (rule-based, ML-based, etc.) extends this class and adds its own
/// detection method while reusing common threshold and scoring logic.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class TextToxicityDetectorBase<T> : TextSafetyModuleBase<T>, ITextToxicityDetector<T>
{
    /// <summary>
    /// The toxicity threshold above which content is flagged.
    /// </summary>
    protected readonly double Threshold;

    /// <summary>
    /// Initializes the toxicity detector base with a threshold.
    /// </summary>
    /// <param name="threshold">The toxicity threshold (0.0 to 1.0). Default: 0.5.</param>
    protected TextToxicityDetectorBase(double threshold = 0.5)
    {
        Threshold = threshold;
    }

    /// <inheritdoc />
    public abstract double GetToxicityScore(string text);
}
