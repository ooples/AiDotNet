using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace AiDotNet.Validation;

/// <summary>
/// Provides concise argument validation methods that throw standard .NET exceptions.
/// </summary>
/// <remarks>
/// <para>
/// Guard methods are the standard way to validate constructor parameters and method arguments
/// throughout AiDotNet. They replace the verbose <c>?? throw new ArgumentNullException</c> pattern
/// with a cleaner, consistent API.
/// </para>
/// <para><b>For Beginners:</b> When you write a class that accepts parameters in its constructor,
/// you need to make sure those parameters aren't null (or invalid). Guard methods are simple
/// one-liners that check a value and throw an exception with a helpful message if it's bad.
///
/// Instead of writing:
/// <code>
/// _model = model ?? throw new ArgumentNullException(nameof(model));
/// </code>
/// You write:
/// <code>
/// Guard.NotNull(model);
/// _model = model;
/// </code>
///
/// On .NET 6+, the parameter name is captured automatically. On .NET Framework 4.7.1,
/// pass <c>nameof(param)</c> explicitly for the best error messages.
/// </para>
/// </remarks>
public static class Guard
{
    /// <summary>
    /// Throws <see cref="ArgumentNullException"/> if <paramref name="value"/> is <c>null</c>.
    /// </summary>
    /// <typeparam name="T">The type of the value being checked.</typeparam>
    /// <param name="value">The value to check for null.</param>
    /// <param name="parameterName">
    /// The name of the parameter. On .NET 6+ this is auto-filled by the compiler.
    /// On .NET Framework, pass <c>nameof(param)</c> explicitly.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="value"/> is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this at the top of constructors for parameters that get
    /// stored in fields. It ensures the value is not null before you use it.
    /// </para>
    /// </remarks>
    public static void NotNull<T>(
        [NotNull] T? value,
        [CallerArgumentExpression(nameof(value))] string? parameterName = null)
        where T : class
    {
        if (value is null)
        {
            throw new ArgumentNullException(parameterName);
        }
    }

    /// <summary>
    /// Throws <see cref="ArgumentNullException"/> if <paramref name="value"/> is <c>null</c>,
    /// or <see cref="ArgumentException"/> if it is empty.
    /// </summary>
    /// <param name="value">The string value to check.</param>
    /// <param name="parameterName">
    /// The name of the parameter. On .NET 6+ this is auto-filled by the compiler.
    /// On .NET Framework, pass <c>nameof(param)</c> explicitly.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="value"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="value"/> is empty.</exception>
    public static void NotNullOrEmpty(
        [NotNull] string? value,
        [CallerArgumentExpression(nameof(value))] string? parameterName = null)
    {
        if (value is null)
        {
            throw new ArgumentNullException(parameterName);
        }

        if (value.Length == 0)
        {
            throw new ArgumentException("Value cannot be empty.", parameterName);
        }
    }

    /// <summary>
    /// Throws <see cref="ArgumentNullException"/> if <paramref name="value"/> is <c>null</c>,
    /// or <see cref="ArgumentException"/> if it is empty or consists only of white-space characters.
    /// </summary>
    /// <param name="value">The string value to check.</param>
    /// <param name="parameterName">
    /// The name of the parameter. On .NET 6+ this is auto-filled by the compiler.
    /// On .NET Framework, pass <c>nameof(param)</c> explicitly.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="value"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="value"/> is empty or whitespace.</exception>
    public static void NotNullOrWhiteSpace(
        [NotNull] string? value,
        [CallerArgumentExpression(nameof(value))] string? parameterName = null)
    {
        if (value is null)
        {
            throw new ArgumentNullException(parameterName);
        }

        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException("Value cannot be empty or whitespace.", parameterName);
        }
    }

    /// <summary>
    /// Throws <see cref="ArgumentOutOfRangeException"/> if <paramref name="value"/> is not positive (less than or equal to zero).
    /// </summary>
    /// <param name="value">The integer value to check.</param>
    /// <param name="parameterName">
    /// The name of the parameter. On .NET 6+ this is auto-filled by the compiler.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="value"/> is &lt;= 0.</exception>
    public static void Positive(
        int value,
        [CallerArgumentExpression(nameof(value))] string? parameterName = null)
    {
        if (value <= 0)
        {
            throw new ArgumentOutOfRangeException(parameterName, value, "Value must be positive.");
        }
    }

    /// <summary>
    /// Throws <see cref="ArgumentOutOfRangeException"/> if <paramref name="value"/> is not positive,
    /// is NaN, or is infinity.
    /// </summary>
    /// <param name="value">The double value to check.</param>
    /// <param name="parameterName">
    /// The name of the parameter. On .NET 6+ this is auto-filled by the compiler.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="value"/> is &lt;= 0, NaN, or infinity.</exception>
    public static void Positive(
        double value,
        [CallerArgumentExpression(nameof(value))] string? parameterName = null)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value <= 0.0)
        {
            throw new ArgumentOutOfRangeException(parameterName, value, "Value must be a positive finite number.");
        }
    }

    /// <summary>
    /// Throws <see cref="ArgumentOutOfRangeException"/> if <paramref name="value"/> is negative (less than zero).
    /// </summary>
    /// <param name="value">The integer value to check.</param>
    /// <param name="parameterName">
    /// The name of the parameter. On .NET 6+ this is auto-filled by the compiler.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="value"/> is &lt; 0.</exception>
    public static void NonNegative(
        int value,
        [CallerArgumentExpression(nameof(value))] string? parameterName = null)
    {
        if (value < 0)
        {
            throw new ArgumentOutOfRangeException(parameterName, value, "Value must be non-negative.");
        }
    }

    /// <summary>
    /// Throws <see cref="ArgumentOutOfRangeException"/> if <paramref name="value"/> is negative,
    /// NaN, or infinity.
    /// </summary>
    /// <param name="value">The double value to check.</param>
    /// <param name="parameterName">
    /// The name of the parameter. On .NET 6+ this is auto-filled by the compiler.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="value"/> is &lt; 0, NaN, or infinity.</exception>
    public static void NonNegative(
        double value,
        [CallerArgumentExpression(nameof(value))] string? parameterName = null)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value < 0.0)
        {
            throw new ArgumentOutOfRangeException(parameterName, value, "Value must be a non-negative finite number.");
        }
    }

    /// <summary>
    /// Throws <see cref="ArgumentOutOfRangeException"/> if <paramref name="value"/> is not
    /// within the inclusive range [<paramref name="min"/>, <paramref name="max"/>].
    /// </summary>
    /// <param name="value">The integer value to check.</param>
    /// <param name="min">The minimum allowed value (inclusive).</param>
    /// <param name="max">The maximum allowed value (inclusive).</param>
    /// <param name="parameterName">
    /// The name of the parameter. On .NET 6+ this is auto-filled by the compiler.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="value"/> is outside the range.</exception>
    public static void InRange(
        int value,
        int min,
        int max,
        [CallerArgumentExpression(nameof(value))] string? parameterName = null)
    {
        if (value < min || value > max)
        {
            throw new ArgumentOutOfRangeException(parameterName, value,
                $"Value must be between {min} and {max} (inclusive).");
        }
    }

    /// <summary>
    /// Throws <see cref="ArgumentOutOfRangeException"/> if <paramref name="value"/> is not
    /// within the inclusive range [<paramref name="min"/>, <paramref name="max"/>],
    /// or if it is NaN or infinity.
    /// </summary>
    /// <param name="value">The double value to check.</param>
    /// <param name="min">The minimum allowed value (inclusive).</param>
    /// <param name="max">The maximum allowed value (inclusive).</param>
    /// <param name="parameterName">
    /// The name of the parameter. On .NET 6+ this is auto-filled by the compiler.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="value"/> is outside the range, NaN, or infinity.</exception>
    public static void InRange(
        double value,
        double min,
        double max,
        [CallerArgumentExpression(nameof(value))] string? parameterName = null)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value < min || value > max)
        {
            throw new ArgumentOutOfRangeException(parameterName, value,
                $"Value must be a finite number between {min} and {max} (inclusive).");
        }
    }
}
