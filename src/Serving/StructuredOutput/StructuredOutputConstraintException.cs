using System;

namespace AiDotNet.Serving.StructuredOutput;

/// <summary>
/// Thrown by an <see cref="ITokenConstraint"/> during masking when generation has reached a state from
/// which the required format can no longer be satisfied — no valid non-EOS continuation remains and the
/// text so far is not a complete valid instance (a non-accepting dead-end).
/// </summary>
/// <remarks>
/// The serving engine catches this per-sequence and fails just that sequence with an error, rather than
/// opening the end-of-sequence token to fake a successful stop. Failing closed guarantees a structured
/// request never returns output that violates its own constraint.
/// </remarks>
public sealed class StructuredOutputConstraintException : Exception
{
    /// <summary>Creates the exception with a message describing the unsatisfiable state.</summary>
    public StructuredOutputConstraintException(string message) : base(message)
    {
    }

    /// <summary>Creates the exception with a message and an inner cause.</summary>
    public StructuredOutputConstraintException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
