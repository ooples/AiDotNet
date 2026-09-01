using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>One bounded diagnostic attached to an evaluation.</summary>
public sealed class EvolutionDiagnostic
{
    /// <summary>Initializes a diagnostic.</summary>
    /// <param name="code">A stable machine-readable code.</param>
    /// <param name="message">A bounded human-readable message.</param>
    /// <param name="isRedacted">Whether sensitive detail was removed.</param>
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
