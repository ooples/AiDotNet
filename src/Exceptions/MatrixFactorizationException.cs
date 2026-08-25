namespace AiDotNet.Exceptions;

/// <summary>
/// Thrown when a matrix factorization cannot proceed because a required pivot is singular or
/// non-finite.
/// </summary>
public sealed class MatrixFactorizationException : AiDotNetException
{
    /// <summary>Initializes the exception with a diagnostic message.</summary>
    /// <param name="message">The message describing the failed factorization.</param>
    public MatrixFactorizationException(string message)
        : base(message)
    {
    }
}
