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
    /// <summary>Initializes a diagnostic.</summary>
    /// <param name="code">A stable machine-readable code.</param>
    /// <param name="message">A bounded human-readable message.</param>
    /// <param name="isRedacted">Whether sensitive detail was removed.</param>
    /// <exception cref="ArgumentNullException"><paramref name="code"/> or <paramref name="message"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="code"/> is empty or whitespace, exceeds 128 characters, or <paramref name="message"/> exceeds
    /// 4096 characters.
    /// </exception>
    public EvolutionDiagnostic(string code, string message, bool isRedacted = false)
    {
        Guard.NotNullOrWhiteSpace(code);
        Guard.NotNull(message);
        if (code.Length > 128) throw new ArgumentException("Diagnostic codes cannot exceed 128 characters.", nameof(code));
        if (message.Length > 4096) throw new ArgumentException("Diagnostic messages cannot exceed 4096 characters.", nameof(message));
        Code = code.Trim();
        Message = message;
        IsRedacted = isRedacted;
    }

    /// <summary>Gets the stable diagnostic code.</summary>
    public string Code { get; }

    /// <summary>Gets the human-readable diagnostic message.</summary>
    public string Message { get; }

    /// <summary>Gets whether sensitive content was redacted.</summary>
    public bool IsRedacted { get; }
}
