namespace AiDotNet.Helpers;

/// <summary>
/// Shared validation helpers for numeric type constraints.
/// </summary>
internal static class NumericGuard
{
    /// <summary>
    /// Throws <see cref="NotSupportedException"/> when <typeparamref name="T"/> is an integer type
    /// that cannot support the fractional arithmetic required by the calling algorithm.
    /// </summary>
    /// <typeparam name="T">The numeric type to check.</typeparam>
    /// <param name="algorithmName">Name of the algorithm for the error message.</param>
    /// <exception cref="NotSupportedException">
    /// Thrown when <typeparamref name="T"/> is an integer type (byte, sbyte, short, ushort, int, uint, long, ulong).
    /// </exception>
    public static void RejectIntegerTypes<T>(string algorithmName)
    {
        var typeCode = Type.GetTypeCode(typeof(T));
        if (typeCode is TypeCode.Int32 or TypeCode.Int64 or TypeCode.UInt32 or TypeCode.UInt64
            or TypeCode.Int16 or TypeCode.UInt16 or TypeCode.Byte or TypeCode.SByte)
        {
            throw new NotSupportedException(
                $"{algorithmName} does not support integer type '{typeof(T).Name}'. " +
                "Use a floating-point type such as float or double.");
        }
    }
}
