using System.Collections.ObjectModel;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>One bounded diagnostic attached to an evaluation.</summary>
/// <remarks>
/// <para>
/// Diagnostics travel with <see cref="EvolutionEvaluation"/> and <see cref="EvolutionTaskResult"/> instances, are
/// retained by the engine for failure-like evaluations (bounded by
/// <see cref="AiDotNet.Configuration.EvolutionEngineOptions.MaxRetainedFailures"/>), are written into checkpoints,
/// and are folded into the run's deterministic state hash. To keep checkpoints small and hashes stable the type
/// enforces hard limits: <see cref="Code"/> is trimmed and may not exceed 128 characters, and <see cref="Message"/>
/// may not exceed 4096. <see cref="IsRedacted"/> records that sensitive detail was removed before the message was
/// stored; the engine itself sets it when it converts an evaluator exception into a diagnostic that names only the
/// exception type.
/// </para>
/// <para>
/// <see cref="Data"/> carries the machine-readable context a caller would otherwise have to parse out of
/// <see cref="Message"/>: the cascade stage that failed, the attempt number, the configured timeout, and so on. It is
/// bounded to <see cref="MaximumDataEntries"/> entries with short keys and values, and it deliberately holds no
/// wall-clock timestamps or elapsed durations, because the whole diagnostic is folded into the run's deterministic
/// state hash and two identical runs must hash identically however long they took. OpenEvolve's equivalent error
/// context is an unbounded dictionary that includes <c>time.time()</c> (evaluator.py:644-666).
/// </para>
/// <para><b>For Beginners:</b> When an evaluator scores a candidate, things can go wrong or be worth noting: a
/// timeout, an exception, an exhausted budget, or a warning about the candidate. A diagnostic is one such note with
/// two parts: a short stable <see cref="Code"/> such as <c>"evaluation_timeout"</c> that programs can match on, and a
/// human-readable <see cref="Message"/> that explains it. Think of one line in a build log, where the error number is
/// the code and the text after it is the message. The size limits exist so that a noisy evaluator cannot bloat your
/// checkpoints, and the redaction flag lets you see at a glance whether a message was trimmed for safety before you
/// forward it to a log or a user.</para>
/// </remarks>
public sealed class EvolutionDiagnostic
{
    /// <summary>The largest number of structured <see cref="Data"/> entries a diagnostic may carry.</summary>
    public const int MaximumDataEntries = 16;

    /// <summary>The largest permitted length of a structured <see cref="Data"/> key.</summary>
    public const int MaximumDataKeyLength = 64;

    /// <summary>The largest permitted length of a structured <see cref="Data"/> value.</summary>
    public const int MaximumDataValueLength = 256;

    private readonly ReadOnlyDictionary<string, string> _data;

    /// <summary>Initializes a diagnostic.</summary>
    /// <param name="code">A stable machine-readable code.</param>
    /// <param name="message">A bounded human-readable message.</param>
    /// <param name="isRedacted">Whether sensitive detail was removed.</param>
    /// <param name="data">
    /// Optional structured context, at most <see cref="MaximumDataEntries"/> entries with keys of at most
    /// <see cref="MaximumDataKeyLength"/> and values of at most <see cref="MaximumDataValueLength"/> characters.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="code"/> or <paramref name="message"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="code"/> is empty or whitespace, exceeds 128 characters, <paramref name="message"/> exceeds
    /// 4096 characters, or <paramref name="data"/> violates its entry, key, or value bounds.
    /// </exception>
    public EvolutionDiagnostic(string code, string message, bool isRedacted = false,
        IReadOnlyDictionary<string, string>? data = null)
    {
        Guard.NotNullOrWhiteSpace(code);
        Guard.NotNull(message);
        if (code.Length > 128) throw new ArgumentException("Diagnostic codes cannot exceed 128 characters.", nameof(code));
        if (message.Length > 4096) throw new ArgumentException("Diagnostic messages cannot exceed 4096 characters.", nameof(message));
        Code = code.Trim();
        Message = message;
        IsRedacted = isRedacted;
        _data = new ReadOnlyDictionary<string, string>(CopyData(data));
    }

    /// <summary>Gets the stable diagnostic code.</summary>
    public string Code { get; }

    /// <summary>Gets the human-readable diagnostic message.</summary>
    public string Message { get; }

    /// <summary>Gets whether sensitive content was redacted.</summary>
    public bool IsRedacted { get; }

    /// <summary>Gets bounded structured context, ordered by key; empty when the diagnostic carries none.</summary>
    public IReadOnlyDictionary<string, string> Data => _data;

    private static Dictionary<string, string> CopyData(IReadOnlyDictionary<string, string>? data)
    {
        var copy = new Dictionary<string, string>(StringComparer.Ordinal);
        if (data is null) return copy;
        if (data.Count > MaximumDataEntries)
            throw new ArgumentException($"A diagnostic may carry at most {MaximumDataEntries} structured entries.", nameof(data));
        foreach (KeyValuePair<string, string> entry in data)
        {
            Guard.NotNullOrWhiteSpace(entry.Key);
            if (entry.Value is null) throw new ArgumentException("Diagnostic data values cannot be null.", nameof(data));
            if (entry.Key.Length > MaximumDataKeyLength)
                throw new ArgumentException($"Diagnostic data keys cannot exceed {MaximumDataKeyLength} characters.", nameof(data));
            if (entry.Value.Length > MaximumDataValueLength)
                throw new ArgumentException($"Diagnostic data values cannot exceed {MaximumDataValueLength} characters.", nameof(data));
            string key = entry.Key.Trim();
            if (copy.ContainsKey(key)) throw new ArgumentException("Diagnostic data keys must be unique.", nameof(data));
            copy.Add(key, entry.Value);
        }
        return copy;
    }
}
