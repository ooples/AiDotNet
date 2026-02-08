namespace AiDotNet.Agents;

/// <summary>
/// Result of validating a hyperparameter value against its definition's constraints.
/// </summary>
internal class HyperparameterValidationResult
{
    /// <summary>
    /// Gets whether the value is valid (correct type, within range).
    /// </summary>
    public bool IsValid { get; init; } = true;

    /// <summary>
    /// Gets whether the value triggered a warning (e.g., at the edge of valid range).
    /// </summary>
    public bool HasWarning { get; init; }

    /// <summary>
    /// Gets the warning message, if any.
    /// </summary>
    public string? Warning { get; init; }

    /// <summary>
    /// Creates a successful validation result with no warnings.
    /// </summary>
    public static HyperparameterValidationResult Valid() => new() { IsValid = true };

    /// <summary>
    /// Creates a validation result with a warning.
    /// </summary>
    public static HyperparameterValidationResult WithWarning(string warning) => new()
    {
        IsValid = true,
        HasWarning = true,
        Warning = warning
    };

    /// <summary>
    /// Creates a failed validation result.
    /// </summary>
    public static HyperparameterValidationResult Invalid(string reason) => new()
    {
        IsValid = false,
        HasWarning = true,
        Warning = reason
    };
}
