#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Operators;

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for double precision.
/// </summary>
public readonly struct SubtractOperatorDouble : IBinaryOperator<double, double>
{
    public double Invoke(double x, double y) => x - y;

#if NET5_0_OR_GREATER
    public Vector128<double> Invoke(Vector128<double> x, Vector128<double> y)
        => Vector128.Subtract(x, y);

    public Vector256<double> Invoke(Vector256<double> x, Vector256<double> y)
        => Vector256.Subtract(x, y);

    public Vector512<double> Invoke(Vector512<double> x, Vector512<double> y)
        => Vector512.Subtract(x, y);
#endif
}

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for single precision.
/// </summary>
public readonly struct SubtractOperatorFloat : IBinaryOperator<float, float>
{
    public float Invoke(float x, float y) => x - y;

#if NET5_0_OR_GREATER
    public Vector128<float> Invoke(Vector128<float> x, Vector128<float> y)
        => Vector128.Subtract(x, y);

    public Vector256<float> Invoke(Vector256<float> x, Vector256<float> y)
        => Vector256.Subtract(x, y);

    public Vector512<float> Invoke(Vector512<float> x, Vector512<float> y)
        => Vector512.Subtract(x, y);
#endif
}

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for integers.
/// </summary>
public readonly struct SubtractOperatorInt : IBinaryOperator<int, int>
{
    public int Invoke(int x, int y) => x - y;

#if NET5_0_OR_GREATER
    public Vector128<int> Invoke(Vector128<int> x, Vector128<int> y)
        => Vector128.Subtract(x, y);

    public Vector256<int> Invoke(Vector256<int> x, Vector256<int> y)
        => Vector256.Subtract(x, y);

    public Vector512<int> Invoke(Vector512<int> x, Vector512<int> y)
        => Vector512.Subtract(x, y);
#endif
}

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for long integers.
/// </summary>
public readonly struct SubtractOperatorLong : IBinaryOperator<long, long>
{
    public long Invoke(long x, long y) => x - y;

#if NET5_0_OR_GREATER
    public Vector128<long> Invoke(Vector128<long> x, Vector128<long> y)
        => Vector128.Subtract(x, y);

    public Vector256<long> Invoke(Vector256<long> x, Vector256<long> y)
        => Vector256.Subtract(x, y);

    public Vector512<long> Invoke(Vector512<long> x, Vector512<long> y)
        => Vector512.Subtract(x, y);
#endif
}

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for short integers.
/// </summary>
public readonly struct SubtractOperatorShort : IBinaryOperator<short, short>
{
    public short Invoke(short x, short y) => (short)(x - y);

#if NET5_0_OR_GREATER
    public Vector128<short> Invoke(Vector128<short> x, Vector128<short> y)
        => Vector128.Subtract(x, y);

    public Vector256<short> Invoke(Vector256<short> x, Vector256<short> y)
        => Vector256.Subtract(x, y);

    public Vector512<short> Invoke(Vector512<short> x, Vector512<short> y)
        => Vector512.Subtract(x, y);
#endif
}

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for unsigned short integers.
/// </summary>
public readonly struct SubtractOperatorUShort : IBinaryOperator<ushort, ushort>
{
    public ushort Invoke(ushort x, ushort y) => (ushort)(x - y);

#if NET5_0_OR_GREATER
    public Vector128<ushort> Invoke(Vector128<ushort> x, Vector128<ushort> y)
        => Vector128.Subtract(x, y);

    public Vector256<ushort> Invoke(Vector256<ushort> x, Vector256<ushort> y)
        => Vector256.Subtract(x, y);

    public Vector512<ushort> Invoke(Vector512<ushort> x, Vector512<ushort> y)
        => Vector512.Subtract(x, y);
#endif
}

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for unsigned integers.
/// </summary>
public readonly struct SubtractOperatorUInt : IBinaryOperator<uint, uint>
{
    public uint Invoke(uint x, uint y) => x - y;

#if NET5_0_OR_GREATER
    public Vector128<uint> Invoke(Vector128<uint> x, Vector128<uint> y)
        => Vector128.Subtract(x, y);

    public Vector256<uint> Invoke(Vector256<uint> x, Vector256<uint> y)
        => Vector256.Subtract(x, y);

    public Vector512<uint> Invoke(Vector512<uint> x, Vector512<uint> y)
        => Vector512.Subtract(x, y);
#endif
}

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for unsigned long integers.
/// </summary>
public readonly struct SubtractOperatorULong : IBinaryOperator<ulong, ulong>
{
    public ulong Invoke(ulong x, ulong y) => x - y;

#if NET5_0_OR_GREATER
    public Vector128<ulong> Invoke(Vector128<ulong> x, Vector128<ulong> y)
        => Vector128.Subtract(x, y);

    public Vector256<ulong> Invoke(Vector256<ulong> x, Vector256<ulong> y)
        => Vector256.Subtract(x, y);

    public Vector512<ulong> Invoke(Vector512<ulong> x, Vector512<ulong> y)
        => Vector512.Subtract(x, y);
#endif
}

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for bytes.
/// </summary>
public readonly struct SubtractOperatorByte : IBinaryOperator<byte, byte>
{
    public byte Invoke(byte x, byte y) => (byte)(x - y);

#if NET5_0_OR_GREATER
    public Vector128<byte> Invoke(Vector128<byte> x, Vector128<byte> y)
        => Vector128.Subtract(x, y);

    public Vector256<byte> Invoke(Vector256<byte> x, Vector256<byte> y)
        => Vector256.Subtract(x, y);

    public Vector512<byte> Invoke(Vector512<byte> x, Vector512<byte> y)
        => Vector512.Subtract(x, y);
#endif
}

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for signed bytes.
/// </summary>
public readonly struct SubtractOperatorSByte : IBinaryOperator<sbyte, sbyte>
{
    public sbyte Invoke(sbyte x, sbyte y) => (sbyte)(x - y);

#if NET5_0_OR_GREATER
    public Vector128<sbyte> Invoke(Vector128<sbyte> x, Vector128<sbyte> y)
        => Vector128.Subtract(x, y);

    public Vector256<sbyte> Invoke(Vector256<sbyte> x, Vector256<sbyte> y)
        => Vector256.Subtract(x, y);

    public Vector512<sbyte> Invoke(Vector512<sbyte> x, Vector512<sbyte> y)
        => Vector512.Subtract(x, y);
#endif
}
