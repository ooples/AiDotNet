using System.Numerics;

namespace AiDotNet.JitCompiler.IR;

/// <summary>
/// Represents the data type of a tensor in the IR.
/// </summary>
public enum IRType
{
    Float32,
    Float64,
    Int32,
    Int64,
    Byte,
    SByte,
    Int16,
    UInt16,
    UInt32,
    UInt64,
    Decimal,
    Half,
    Complex
}

/// <summary>
/// Helper methods for IRType.
/// </summary>
public static class IRTypeExtensions
{
    /// <summary>
    /// Gets the IRType for a given System.Type.
    /// </summary>
    /// <param name="type">The System.Type to convert.</param>
    /// <returns>The corresponding IRType.</returns>
    /// <exception cref="ArgumentNullException">Thrown when type is null.</exception>
    /// <exception cref="NotSupportedException">Thrown when the type is not supported in IR.</exception>
    public static IRType FromSystemType(Type type)
    {
        if (type == null) throw new ArgumentNullException(nameof(type));

        return type switch
        {
            Type t when t == typeof(float) => IRType.Float32,
            Type t when t == typeof(double) => IRType.Float64,
            Type t when t == typeof(int) => IRType.Int32,
            Type t when t == typeof(long) => IRType.Int64,
            Type t when t == typeof(byte) => IRType.Byte,
            Type t when t == typeof(sbyte) => IRType.SByte,
            Type t when t == typeof(short) => IRType.Int16,
            Type t when t == typeof(ushort) => IRType.UInt16,
            Type t when t == typeof(uint) => IRType.UInt32,
            Type t when t == typeof(ulong) => IRType.UInt64,
            Type t when t == typeof(decimal) => IRType.Decimal,
            Type t when t == typeof(Half) => IRType.Half,
            Type t when t == typeof(Complex) => IRType.Complex,
            _ => throw new NotSupportedException($"Type {type} not supported in IR")
        };
    }

    /// <summary>
    /// Gets the System.Type for a given IRType.
    /// </summary>
    public static Type ToSystemType(this IRType irType)
    {
        return irType switch
        {
            IRType.Float32 => typeof(float),
            IRType.Float64 => typeof(double),
            IRType.Int32 => typeof(int),
            IRType.Int64 => typeof(long),
            IRType.Byte => typeof(byte),
            IRType.SByte => typeof(sbyte),
            IRType.Int16 => typeof(short),
            IRType.UInt16 => typeof(ushort),
            IRType.UInt32 => typeof(uint),
            IRType.UInt64 => typeof(ulong),
            IRType.Decimal => typeof(decimal),
            IRType.Half => typeof(Half),
            IRType.Complex => typeof(Complex),
            _ => throw new NotSupportedException($"IRType {irType} conversion not supported")
        };
    }
}
