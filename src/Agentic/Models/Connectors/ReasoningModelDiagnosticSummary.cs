using System.Globalization;
using AiDotNet.Validation;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>One distinct request adjustment together with how often it occurred.</summary>
/// <remarks>
/// <para>
/// A run that makes ten thousand calls against a reasoning model drops the same setting ten thousand times. Listing
/// each occurrence tells a reader nothing the first one did not, so the collecting sink groups them: this type
/// pairs the first record seen for a given adjustment with the number of times that adjustment recurred.
/// </para>
/// <para><b>For Beginners:</b> Instead of ten thousand identical notices saying "temperature was removed", you get
/// one notice and the number 10000. This is that pair.</para>
/// </remarks>
public sealed class ReasoningModelDiagnosticSummary
{
    /// <summary>Initializes a summary entry.</summary>
    /// <param name="diagnostic">The representative record for this adjustment.</param>
    /// <param name="occurrences">How many times the adjustment occurred; must be positive.</param>
    /// <exception cref="ArgumentNullException"><paramref name="diagnostic"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="occurrences"/> is not positive.</exception>
    public ReasoningModelDiagnosticSummary(ReasoningModelDiagnostic diagnostic, long occurrences)
    {
        Guard.NotNull(diagnostic);
        if (occurrences <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(occurrences), occurrences, "Value must be positive.");
        }

        Diagnostic = diagnostic;
        Occurrences = occurrences;
    }

    /// <summary>Gets the representative record for this adjustment.</summary>
    public ReasoningModelDiagnostic Diagnostic { get; }

    /// <summary>Gets how many times the adjustment occurred.</summary>
    public long Occurrences { get; }

    /// <summary>Returns the adjustment and its count.</summary>
    /// <returns>A short description carrying no prompt text, endpoint, or credential.</returns>
    public override string ToString() => string.Format(
        CultureInfo.InvariantCulture,
        "{0} x{1}",
        Diagnostic,
        Occurrences);
}
