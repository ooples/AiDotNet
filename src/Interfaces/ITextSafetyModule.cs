using AiDotNet.Safety;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for safety modules that operate on text content.
/// </summary>
/// <remarks>
/// <para>
/// Text safety modules analyze string content for safety risks such as toxicity,
/// PII exposure, jailbreak attempts, hallucinations, and copyright violations.
/// They extend <see cref="ISafetyModule{T}"/> with text-specific evaluation methods.
/// </para>
/// <para>
/// <b>For Beginners:</b> Text safety modules check written content for problems.
/// Unlike the base ISafetyModule which works on numeric vectors, text safety modules
/// can accept raw strings directly, making them easier to use for text-based applications.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ITextSafetyModule<T> : ISafetyModule<T>
{
    /// <summary>
    /// Evaluates the given text string for safety and returns any findings.
    /// </summary>
    /// <param name="text">The text content to evaluate.</param>
    /// <returns>
    /// A list of safety findings. An empty list means no safety issues were detected.
    /// </returns>
    IReadOnlyList<SafetyFinding> EvaluateText(string text);
}
