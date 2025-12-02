using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Operators;

/// <summary>
/// Implements element-wise division using hardware-accelerated SIMD instructions for double precision.
/// </summary>
public readonly struct DivideOperatorDouble : IBinaryOperator<double, double>
{
    public double Invoke(double x, double y) => x / y;

#if NET5_0_OR_GREATER
    public Vector128<double> Invoke(Vector128<double> x, Vector128<double> y)
        => Vector128.Divide(x, y);

    public Vector256<double> Invoke(Vector256<double> x, Vector256<double> y)
        => Vector256.Divide(x, y);

    public Vector512<double> Invoke(Vector512<double> x, Vector512<double> y)
        => Vector512.Divide(x, y);
#endif
}

/// <summary>
/// Implements element-wise division using hardware-accelerated SIMD instructions for single precision.
/// </summary>
public readonly struct DivideOperatorFloat : IBinaryOperator<float, float>
{
    public float Invoke(float x, float y) => x / y;

#if NET5_0_OR_GREATER
    public Vector128<float> Invoke(Vector128<float> x, Vector128<float> y)
        => Vector128.Divide(x, y);

    public Vector256<float> Invoke(Vector256<float> x, Vector256<float> y)
        => Vector256.Divide(x, y);

    public Vector512<float> Invoke(Vector512<float> x, Vector512<float> y)
        => Vector512.Divide(x, y);
#endif
}

/// <summary>
/// Implements element-wise division for integers.
/// </summary>
/// <remarks>
/// Integer division doesn't have direct SIMD support, so this falls back to scalar operations
/// within the vector processing loop for optimal cache utilization.
/// </remarks>
public readonly struct DivideOperatorInt : IBinaryOperator<int, int>
{
    public int Invoke(int x, int y) => x / y;

#if NET5_0_OR_GREATER
    public Vector128<int> Invoke(Vector128<int> x, Vector128<int> y)
    {
        // Integer division doesn't have direct SIMD support
        Span<int> xValues = stackalloc int[Vector128<int>.Count];
        Span<int> yValues = stackalloc int[Vector128<int>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector128.Create(xValues);
    }

    public Vector256<int> Invoke(Vector256<int> x, Vector256<int> y)
    {
        Span<int> xValues = stackalloc int[Vector256<int>.Count];
        Span<int> yValues = stackalloc int[Vector256<int>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector256.Create(xValues);
    }

    public Vector512<int> Invoke(Vector512<int> x, Vector512<int> y)
    {
        Span<int> xValues = stackalloc int[Vector512<int>.Count];
        Span<int> yValues = stackalloc int[Vector512<int>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector512.Create(xValues);
    }
#endif
}

/// <summary>
/// Implements element-wise division for long integers.
/// </summary>
/// <remarks>
/// Integer division doesn't have direct SIMD support, so this falls back to scalar operations
/// within the vector processing loop for optimal cache utilization.
/// </remarks>
public readonly struct DivideOperatorLong : IBinaryOperator<long, long>
{
    public long Invoke(long x, long y) => x / y;

#if NET5_0_OR_GREATER
    public Vector128<long> Invoke(Vector128<long> x, Vector128<long> y)
    {
        Span<long> xValues = stackalloc long[Vector128<long>.Count];
        Span<long> yValues = stackalloc long[Vector128<long>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector128.Create(xValues);
    }

    public Vector256<long> Invoke(Vector256<long> x, Vector256<long> y)
    {
        Span<long> xValues = stackalloc long[Vector256<long>.Count];
        Span<long> yValues = stackalloc long[Vector256<long>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector256.Create(xValues);
    }

    public Vector512<long> Invoke(Vector512<long> x, Vector512<long> y)
    {
        Span<long> xValues = stackalloc long[Vector512<long>.Count];
        Span<long> yValues = stackalloc long[Vector512<long>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector512.Create(xValues);
    }
#endif
}

/// <summary>
/// Implements element-wise division for short integers.
/// </summary>
public readonly struct DivideOperatorShort : IBinaryOperator<short, short>
{
    public short Invoke(short x, short y) => (short)(x / y);

#if NET5_0_OR_GREATER
    public Vector128<short> Invoke(Vector128<short> x, Vector128<short> y)
    {
        Span<short> xValues = stackalloc short[Vector128<short>.Count];
        Span<short> yValues = stackalloc short[Vector128<short>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] = (short)(xValues[i] / yValues[i]);

        return Vector128.Create(xValues);
    }

    public Vector256<short> Invoke(Vector256<short> x, Vector256<short> y)
    {
        Span<short> xValues = stackalloc short[Vector256<short>.Count];
        Span<short> yValues = stackalloc short[Vector256<short>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] = (short)(xValues[i] / yValues[i]);

        return Vector256.Create(xValues);
    }

    public Vector512<short> Invoke(Vector512<short> x, Vector512<short> y)
    {
        Span<short> xValues = stackalloc short[Vector512<short>.Count];
        Span<short> yValues = stackalloc short[Vector512<short>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] = (short)(xValues[i] / yValues[i]);

        return Vector512.Create(xValues);
    }
#endif
}

/// <summary>
/// Implements element-wise division for unsigned short integers.
/// </summary>
public readonly struct DivideOperatorUShort : IBinaryOperator<ushort, ushort>
{
    public ushort Invoke(ushort x, ushort y) => (ushort)(x / y);

#if NET5_0_OR_GREATER
    public Vector128<ushort> Invoke(Vector128<ushort> x, Vector128<ushort> y)
    {
        Span<ushort> xValues = stackalloc ushort[Vector128<ushort>.Count];
        Span<ushort> yValues = stackalloc ushort[Vector128<ushort>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] = (ushort)(xValues[i] / yValues[i]);

        return Vector128.Create(xValues);
    }

    public Vector256<ushort> Invoke(Vector256<ushort> x, Vector256<ushort> y)
    {
        Span<ushort> xValues = stackalloc ushort[Vector256<ushort>.Count];
        Span<ushort> yValues = stackalloc ushort[Vector256<ushort>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] = (ushort)(xValues[i] / yValues[i]);

        return Vector256.Create(xValues);
    }

    public Vector512<ushort> Invoke(Vector512<ushort> x, Vector512<ushort> y)
    {
        Span<ushort> xValues = stackalloc ushort[Vector512<ushort>.Count];
        Span<ushort> yValues = stackalloc ushort[Vector512<ushort>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] = (ushort)(xValues[i] / yValues[i]);

        return Vector512.Create(xValues);
    }
#endif
}

/// <summary>
/// Implements element-wise division for unsigned integers.
/// </summary>
public readonly struct DivideOperatorUInt : IBinaryOperator<uint, uint>
{
    public uint Invoke(uint x, uint y) => x / y;

#if NET5_0_OR_GREATER
    public Vector128<uint> Invoke(Vector128<uint> x, Vector128<uint> y)
    {
        Span<uint> xValues = stackalloc uint[Vector128<uint>.Count];
        Span<uint> yValues = stackalloc uint[Vector128<uint>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector128.Create(xValues);
    }

    public Vector256<uint> Invoke(Vector256<uint> x, Vector256<uint> y)
    {
        Span<uint> xValues = stackalloc uint[Vector256<uint>.Count];
        Span<uint> yValues = stackalloc uint[Vector256<uint>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector256.Create(xValues);
    }

    public Vector512<uint> Invoke(Vector512<uint> x, Vector512<uint> y)
    {
        Span<uint> xValues = stackalloc uint[Vector512<uint>.Count];
        Span<uint> yValues = stackalloc uint[Vector512<uint>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector512.Create(xValues);
    }
#endif
}

/// <summary>
/// Implements element-wise division for unsigned long integers.
/// </summary>
public readonly struct DivideOperatorULong : IBinaryOperator<ulong, ulong>
{
    public ulong Invoke(ulong x, ulong y) => x / y;

#if NET5_0_OR_GREATER
    public Vector128<ulong> Invoke(Vector128<ulong> x, Vector128<ulong> y)
    {
        Span<ulong> xValues = stackalloc ulong[Vector128<ulong>.Count];
        Span<ulong> yValues = stackalloc ulong[Vector128<ulong>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector128.Create(xValues);
    }

    public Vector256<ulong> Invoke(Vector256<ulong> x, Vector256<ulong> y)
    {
        Span<ulong> xValues = stackalloc ulong[Vector256<ulong>.Count];
        Span<ulong> yValues = stackalloc ulong[Vector256<ulong>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector256.Create(xValues);
    }

    public Vector512<ulong> Invoke(Vector512<ulong> x, Vector512<ulong> y)
    {
        Span<ulong> xValues = stackalloc ulong[Vector512<ulong>.Count];
        Span<ulong> yValues = stackalloc ulong[Vector512<ulong>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector512.Create(xValues);
    }
#endif
}

/// <summary>
/// Implements element-wise division for bytes.
/// </summary>
public readonly struct DivideOperatorByte : IBinaryOperator<byte, byte>
{
    public byte Invoke(byte x, byte y) => (byte)(x / y);

#if NET5_0_OR_GREATER
    public Vector128<byte> Invoke(Vector128<byte> x, Vector128<byte> y)
    {
        Span<byte> xValues = stackalloc byte[Vector128<byte>.Count];
        Span<byte> yValues = stackalloc byte[Vector128<byte>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] = (byte)(xValues[i] / yValues[i]);

        return Vector128.Create(xValues);
    }

    public Vector256<byte> Invoke(Vector256<byte> x, Vector256<byte> y)
    {
        Span<byte> xValues = stackalloc byte[Vector256<byte>.Count];
        Span<byte> yValues = stackalloc byte[Vector256<byte>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] = (byte)(xValues[i] / yValues[i]);

        return Vector256.Create(xValues);
    }

    public Vector512<byte> Invoke(Vector512<byte> x, Vector512<byte> y)
    {
        Span<byte> xValues = stackalloc byte[Vector512<byte>.Count];
        Span<byte> yValues = stackalloc byte[Vector512<byte>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] = (byte)(xValues[i] / yValues[i]);

        return Vector512.Create(xValues);
    }
#endif
}

/// <summary>
/// Implements element-wise division for signed bytes.
/// </summary>
public readonly struct DivideOperatorSByte : IBinaryOperator<sbyte, sbyte>
{
    public sbyte Invoke(sbyte x, sbyte y) => (sbyte)(x / y);

#if NET5_0_OR_GREATER
    public Vector128<sbyte> Invoke(Vector128<sbyte> x, Vector128<sbyte> y)
    {
        Span<sbyte> xValues = stackalloc sbyte[Vector128<sbyte>.Count];
        Span<sbyte> yValues = stackalloc sbyte[Vector128<sbyte>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] = (sbyte)(xValues[i] / yValues[i]);

        return Vector128.Create(xValues);
    }

    public Vector256<sbyte> Invoke(Vector256<sbyte> x, Vector256<sbyte> y)
    {
        Span<sbyte> xValues = stackalloc sbyte[Vector256<sbyte>.Count];
        Span<sbyte> yValues = stackalloc sbyte[Vector256<sbyte>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] = (sbyte)(xValues[i] / yValues[i]);

        return Vector256.Create(xValues);
    }

    public Vector512<sbyte> Invoke(Vector512<sbyte> x, Vector512<sbyte> y)
    {
        Span<sbyte> xValues = stackalloc sbyte[Vector512<sbyte>.Count];
        Span<sbyte> yValues = stackalloc sbyte[Vector512<sbyte>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] = (sbyte)(xValues[i] / yValues[i]);

        return Vector512.Create(xValues);
    }
#endif
}
