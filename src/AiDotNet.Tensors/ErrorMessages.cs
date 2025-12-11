namespace AiDotNet.Tensors;

/// <summary>
/// Centralized error messages for consistent exception handling across the library.
/// </summary>
internal static class ErrorMessages
{
    // Vector/Span length errors
    internal const string VectorsSameLength = "Vectors must have the same length";
    internal const string SpansSameLength = "Spans must have the same length.";
    internal const string AllSpansSameLength = "All spans must have the same length.";
    internal const string InputDestinationSameLength = "Input and destination spans must have the same length.";

    // Empty collection errors
    internal const string VectorCannotBeEmpty = "Vector cannot be empty";
    internal const string SpanCannotBeEmpty = "Span cannot be empty.";
}
