using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>Names one SEARCH/REPLACE block that could not be parsed or applied, and why.</summary>
/// <remarks>
/// <para>
/// This type is the concrete answer to the reference implementation's most damaging silence. OpenEvolve's
/// <c>apply_diff</c> loops over blocks, tries a first-window line match, and simply moves on when there is no
/// match: nothing is logged, the caller sees a program that may be identical to its parent, and the evaluator is
/// still paid for. Every rejection here instead surfaces a typed <see cref="Reason"/>, the failing block's
/// <see cref="BlockOrdinal"/>, and a bounded, control-character-sanitized <see cref="SearchExcerpt"/> that is safe
/// to place in a retry prompt or a log.
/// </para>
/// <para><b>For Beginners:</b> When a model's edit instruction cannot be used, this object explains which
/// instruction failed and what was wrong with it. The excerpt is a short, cleaned-up snippet of the text the model
/// asked to find, trimmed so a huge or unprintable response cannot flood your logs. Feeding these messages back to
/// the model in the next attempt is usually enough for it to correct itself.</para>
/// </remarks>
public sealed class ProgramDiffFailure
{
    /// <summary>Initializes a failure record.</summary>
    /// <param name="reason">The typed reason the block was rejected.</param>
    /// <param name="message">A bounded human-readable explanation.</param>
    /// <param name="blockOrdinal">The zero-based position of the offending block, or -1 when it applies to the whole response.</param>
    /// <param name="searchExcerpt">A bounded, sanitized excerpt of the search text, or an empty string.</param>
    /// <exception cref="ArgumentNullException"><paramref name="message"/> or <paramref name="searchExcerpt"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="message"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="reason"/> is not a defined value, or <paramref name="blockOrdinal"/> is below -1.</exception>
    public ProgramDiffFailure(ProgramDiffFailureReason reason, string message, int blockOrdinal = -1, string searchExcerpt = "")
    {
        if (!Enum.IsDefined(typeof(ProgramDiffFailureReason), reason)) throw new ArgumentOutOfRangeException(nameof(reason));
        Guard.NotNullOrWhiteSpace(message);
        Guard.NotNull(searchExcerpt);
        if (blockOrdinal < -1) throw new ArgumentOutOfRangeException(nameof(blockOrdinal));
        Reason = reason;
        Message = message;
        BlockOrdinal = blockOrdinal;
        SearchExcerpt = searchExcerpt;
    }

    /// <summary>Gets the typed reason the block was rejected.</summary>
    public ProgramDiffFailureReason Reason { get; }

    /// <summary>Gets the bounded human-readable explanation.</summary>
    public string Message { get; }

    /// <summary>Gets the zero-based ordinal of the offending block, or -1 when the failure covers the whole response.</summary>
    public int BlockOrdinal { get; }

    /// <summary>Gets a bounded, sanitized excerpt of the search text that failed, or an empty string.</summary>
    public string SearchExcerpt { get; }

    /// <summary>Converts this failure into an engine diagnostic suitable for an evaluation result.</summary>
    /// <returns>
    /// A diagnostic whose code is the lowercase reason name and whose message is <see cref="Message"/>, flagged as
    /// redacted because the excerpt it carries was truncated and sanitized.
    /// </returns>
    public EvolutionDiagnostic ToDiagnostic() => new(
        "program_diff_" + Reason.ToString().ToLowerInvariant(),
        ProgramText.Bound(Message, 4096),
        isRedacted: true);

    /// <summary>Returns the reason, ordinal, and message without echoing raw program text.</summary>
    /// <returns>A short diagnostic label for this failure.</returns>
    public override string ToString() => string.Concat(
        Reason.ToString(), " (block ",
        BlockOrdinal.ToString(System.Globalization.CultureInfo.InvariantCulture), "): ", Message);
}
