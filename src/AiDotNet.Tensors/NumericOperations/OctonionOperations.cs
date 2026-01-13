using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Provides numeric operations for Octonion{T}.
/// </summary>
public class OctonionOperations<T> : INumericOperations<Octonion<T>>
{
    private readonly INumericOperations<T> _ops;

    public OctonionOperations()
    {
        _ops = MathHelper.GetNumericOperations<T>();
    }

    public Octonion<T> Add(Octonion<T> a, Octonion<T> b) => a + b;

    public Octonion<T> Subtract(Octonion<T> a, Octonion<T> b) => a - b;

    public Octonion<T> Multiply(Octonion<T> a, Octonion<T> b) => a * b;

    public Octonion<T> Divide(Octonion<T> a, Octonion<T> b) => a / b;

    public Octonion<T> Negate(Octonion<T> a)
        => new Octonion<T>(
            _ops.Negate(a.Scalar),
            _ops.Negate(a.E1),
            _ops.Negate(a.E2),
            _ops.Negate(a.E3),
            _ops.Negate(a.E4),
            _ops.Negate(a.E5),
            _ops.Negate(a.E6),
            _ops.Negate(a.E7));

    public Octonion<T> Zero => new Octonion<T>(
        _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero);

    public Octonion<T> One => new Octonion<T>(
        _ops.One, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero);

    public Octonion<T> Sqrt(Octonion<T> value)
    {
        T magnitude = value.Magnitude;
        if (_ops.Equals(magnitude, _ops.Zero))
            return Zero;

        T half = _ops.FromDouble(0.5);
        T scalarTerm = _ops.Sqrt(_ops.Multiply(half, _ops.Add(magnitude, value.Scalar)));

        T vectorMagnitude = value.VectorMagnitude;
        if (_ops.Equals(vectorMagnitude, _ops.Zero))
        {
            return new Octonion<T>(
                scalarTerm, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero);
        }

        T vectorTerm = _ops.Sqrt(_ops.Multiply(half, _ops.Subtract(magnitude, value.Scalar)));
        T scale = _ops.Divide(vectorTerm, vectorMagnitude);

        return new Octonion<T>(
            scalarTerm,
            _ops.Multiply(value.E1, scale),
            _ops.Multiply(value.E2, scale),
            _ops.Multiply(value.E3, scale),
            _ops.Multiply(value.E4, scale),
            _ops.Multiply(value.E5, scale),
            _ops.Multiply(value.E6, scale),
            _ops.Multiply(value.E7, scale));
    }

    public Octonion<T> FromDouble(double value)
        => new Octonion<T>(_ops.FromDouble(value), _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero);

    public int ToInt32(Octonion<T> value)
    {
        double magnitude = Convert.ToDouble(value.Magnitude);
        return (int)Math.Round(magnitude);
    }

    public bool GreaterThan(Octonion<T> a, Octonion<T> b)
        => _ops.GreaterThan(a.Magnitude, b.Magnitude);

    public bool LessThan(Octonion<T> a, Octonion<T> b)
        => _ops.LessThan(a.Magnitude, b.Magnitude);

    public Octonion<T> Abs(Octonion<T> value)
        => new Octonion<T>(value.Magnitude, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero);

    public Octonion<T> Square(Octonion<T> value) => value * value;

    public Octonion<T> Exp(Octonion<T> value)
    {
        T vectorMagnitude = value.VectorMagnitude;
        T expScalar = _ops.Exp(value.Scalar);

        if (_ops.Equals(vectorMagnitude, _ops.Zero))
        {
            return new Octonion<T>(
                expScalar, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero);
        }

        double vMag = Convert.ToDouble(vectorMagnitude);
        T cos = _ops.FromDouble(Math.Cos(vMag));
        T sin = _ops.FromDouble(Math.Sin(vMag));
        T scale = _ops.Divide(sin, vectorMagnitude);
        T vectorScale = _ops.Multiply(expScalar, scale);

        return new Octonion<T>(
            _ops.Multiply(expScalar, cos),
            _ops.Multiply(value.E1, vectorScale),
            _ops.Multiply(value.E2, vectorScale),
            _ops.Multiply(value.E3, vectorScale),
            _ops.Multiply(value.E4, vectorScale),
            _ops.Multiply(value.E5, vectorScale),
            _ops.Multiply(value.E6, vectorScale),
            _ops.Multiply(value.E7, vectorScale));
    }

    public bool Equals(Octonion<T> a, Octonion<T> b) => a == b;

    public Octonion<T> Power(Octonion<T> baseValue, Octonion<T> exponent)
    {
        if (baseValue == Zero && exponent == Zero)
            return One;

        return Exp(Multiply(Log(baseValue), exponent));
    }

    public Octonion<T> Log(Octonion<T> value)
    {
        T magnitude = value.Magnitude;
        T vectorMagnitude = value.VectorMagnitude;
        T scalarPart = _ops.Log(magnitude);

        if (_ops.Equals(vectorMagnitude, _ops.Zero))
        {
            return new Octonion<T>(
                scalarPart, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero);
        }

        double ratio = Convert.ToDouble(_ops.Divide(value.Scalar, magnitude));
        // Clamp ratio to [-1, 1] to prevent NaN from floating-point errors
        ratio = Math.Max(-1.0, Math.Min(1.0, ratio));
        double angle = Math.Acos(ratio);
        T scale = _ops.Divide(_ops.FromDouble(angle), vectorMagnitude);

        return new Octonion<T>(
            scalarPart,
            _ops.Multiply(value.E1, scale),
            _ops.Multiply(value.E2, scale),
            _ops.Multiply(value.E3, scale),
            _ops.Multiply(value.E4, scale),
            _ops.Multiply(value.E5, scale),
            _ops.Multiply(value.E6, scale),
            _ops.Multiply(value.E7, scale));
    }

    public bool GreaterThanOrEquals(Octonion<T> a, Octonion<T> b)
        => _ops.GreaterThanOrEquals(a.Magnitude, b.Magnitude);

    public bool LessThanOrEquals(Octonion<T> a, Octonion<T> b)
        => _ops.LessThanOrEquals(a.Magnitude, b.Magnitude);

    public int Compare(Octonion<T> a, Octonion<T> b)
        => _ops.Compare(a.Magnitude, b.Magnitude);

    public Octonion<T> Round(Octonion<T> value)
        => new Octonion<T>(
            _ops.Round(value.Scalar),
            _ops.Round(value.E1),
            _ops.Round(value.E2),
            _ops.Round(value.E3),
            _ops.Round(value.E4),
            _ops.Round(value.E5),
            _ops.Round(value.E6),
            _ops.Round(value.E7));

    public Octonion<T> Floor(Octonion<T> value)
        => new Octonion<T>(
            _ops.Floor(value.Scalar),
            _ops.Floor(value.E1),
            _ops.Floor(value.E2),
            _ops.Floor(value.E3),
            _ops.Floor(value.E4),
            _ops.Floor(value.E5),
            _ops.Floor(value.E6),
            _ops.Floor(value.E7));

    public Octonion<T> Ceiling(Octonion<T> value)
        => new Octonion<T>(
            _ops.Ceiling(value.Scalar),
            _ops.Ceiling(value.E1),
            _ops.Ceiling(value.E2),
            _ops.Ceiling(value.E3),
            _ops.Ceiling(value.E4),
            _ops.Ceiling(value.E5),
            _ops.Ceiling(value.E6),
            _ops.Ceiling(value.E7));

    public Octonion<T> Frac(Octonion<T> value)
        => new Octonion<T>(
            _ops.Frac(value.Scalar),
            _ops.Frac(value.E1),
            _ops.Frac(value.E2),
            _ops.Frac(value.E3),
            _ops.Frac(value.E4),
            _ops.Frac(value.E5),
            _ops.Frac(value.E6),
            _ops.Frac(value.E7));

    public Octonion<T> Sin(Octonion<T> value)
        => new Octonion<T>(
            _ops.Sin(value.Scalar),
            _ops.Sin(value.E1),
            _ops.Sin(value.E2),
            _ops.Sin(value.E3),
            _ops.Sin(value.E4),
            _ops.Sin(value.E5),
            _ops.Sin(value.E6),
            _ops.Sin(value.E7));

    public Octonion<T> Cos(Octonion<T> value)
        => new Octonion<T>(
            _ops.Cos(value.Scalar),
            _ops.Cos(value.E1),
            _ops.Cos(value.E2),
            _ops.Cos(value.E3),
            _ops.Cos(value.E4),
            _ops.Cos(value.E5),
            _ops.Cos(value.E6),
            _ops.Cos(value.E7));

    public Octonion<T> MinValue => new Octonion<T>(
        _ops.MinValue, _ops.MinValue, _ops.MinValue, _ops.MinValue, _ops.MinValue, _ops.MinValue, _ops.MinValue, _ops.MinValue);

    public Octonion<T> MaxValue => new Octonion<T>(
        _ops.MaxValue, _ops.MaxValue, _ops.MaxValue, _ops.MaxValue, _ops.MaxValue, _ops.MaxValue, _ops.MaxValue, _ops.MaxValue);

    public bool IsNaN(Octonion<T> value)
    {
        return _ops.IsNaN(value.Scalar) ||
               _ops.IsNaN(value.E1) ||
               _ops.IsNaN(value.E2) ||
               _ops.IsNaN(value.E3) ||
               _ops.IsNaN(value.E4) ||
               _ops.IsNaN(value.E5) ||
               _ops.IsNaN(value.E6) ||
               _ops.IsNaN(value.E7);
    }

    public bool IsInfinity(Octonion<T> value)
    {
        return _ops.IsInfinity(value.Scalar) ||
               _ops.IsInfinity(value.E1) ||
               _ops.IsInfinity(value.E2) ||
               _ops.IsInfinity(value.E3) ||
               _ops.IsInfinity(value.E4) ||
               _ops.IsInfinity(value.E5) ||
               _ops.IsInfinity(value.E6) ||
               _ops.IsInfinity(value.E7);
    }

    public Octonion<T> SignOrZero(Octonion<T> value)
    {
        T magnitude = value.Magnitude;
        if (_ops.Equals(magnitude, _ops.Zero))
            return Zero;

        T invMagnitude = _ops.Divide(_ops.One, magnitude);
        return value.Scale(invMagnitude);
    }

    public int PrecisionBits => _ops.PrecisionBits;

    public float ToFloat(Octonion<T> value)
    {
        if (!value.IsScalar)
        {
            throw new NotSupportedException(
                "Cannot convert Octonion<T> with non-zero vector component to scalar float. " +
                "Extract Scalar explicitly if this is intentional.");
        }
        return _ops.ToFloat(value.Scalar);
    }

    public Octonion<T> FromFloat(float value)
        => new Octonion<T>(_ops.FromFloat(value), _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero);

    public Half ToHalf(Octonion<T> value)
    {
        if (!value.IsScalar)
        {
            throw new NotSupportedException(
                "Cannot convert Octonion<T> with non-zero vector component to scalar Half. " +
                "Extract Scalar explicitly if this is intentional.");
        }
        return _ops.ToHalf(value.Scalar);
    }

    public Octonion<T> FromHalf(Half value)
        => new Octonion<T>(_ops.FromHalf(value), _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero, _ops.Zero);

    public double ToDouble(Octonion<T> value)
    {
        if (!value.IsScalar)
        {
            throw new NotSupportedException(
                "Cannot convert Octonion<T> with non-zero vector component to scalar double. " +
                "Extract Scalar explicitly if this is intentional.");
        }
        return _ops.ToDouble(value.Scalar);
    }

    public bool SupportsCpuAcceleration => false;

    public bool SupportsGpuAcceleration => false;

    #region IVectorizedOperations<Octonion<T>> fallback implementations

    public void Add(ReadOnlySpan<Octonion<T>> x, ReadOnlySpan<Octonion<T>> y, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Add(this, x, y, destination);

    public void Subtract(ReadOnlySpan<Octonion<T>> x, ReadOnlySpan<Octonion<T>> y, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Subtract(this, x, y, destination);

    public void Multiply(ReadOnlySpan<Octonion<T>> x, ReadOnlySpan<Octonion<T>> y, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Multiply(this, x, y, destination);

    public void Divide(ReadOnlySpan<Octonion<T>> x, ReadOnlySpan<Octonion<T>> y, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Divide(this, x, y, destination);

    public Octonion<T> Dot(ReadOnlySpan<Octonion<T>> x, ReadOnlySpan<Octonion<T>> y)
        => VectorizedOperationsFallback.Dot(this, x, y);

    public Octonion<T> Sum(ReadOnlySpan<Octonion<T>> x)
        => VectorizedOperationsFallback.Sum(this, x);

    public Octonion<T> Max(ReadOnlySpan<Octonion<T>> x)
        => VectorizedOperationsFallback.Max(this, x);

    public Octonion<T> Min(ReadOnlySpan<Octonion<T>> x)
        => VectorizedOperationsFallback.Min(this, x);

    public void Exp(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Exp(this, x, destination);

    public void Log(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Log(this, x, destination);

    public void Tanh(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Tanh(this, x, destination);

    public void Sigmoid(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Sigmoid(this, x, destination);

    public void Log2(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Log2(this, x, destination);

    public void SoftMax(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.SoftMax(this, x, destination);

    public Octonion<T> CosineSimilarity(ReadOnlySpan<Octonion<T>> x, ReadOnlySpan<Octonion<T>> y)
        => VectorizedOperationsFallback.CosineSimilarity(this, x, y);

    public void Fill(Span<Octonion<T>> destination, Octonion<T> value) => destination.Fill(value);

    public void MultiplyScalar(ReadOnlySpan<Octonion<T>> x, Octonion<T> scalar, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.MultiplyScalar(this, x, scalar, destination);

    public void DivideScalar(ReadOnlySpan<Octonion<T>> x, Octonion<T> scalar, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.DivideScalar(this, x, scalar, destination);

    public void AddScalar(ReadOnlySpan<Octonion<T>> x, Octonion<T> scalar, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.AddScalar(this, x, scalar, destination);

    public void SubtractScalar(ReadOnlySpan<Octonion<T>> x, Octonion<T> scalar, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.SubtractScalar(this, x, scalar, destination);

    public void Sqrt(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Sqrt(this, x, destination);

    public void Abs(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Abs(this, x, destination);

    public void Negate(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Negate(this, x, destination);

    public void Clip(ReadOnlySpan<Octonion<T>> x, Octonion<T> min, Octonion<T> max, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Clip(this, x, min, max, destination);

    public void Pow(ReadOnlySpan<Octonion<T>> x, Octonion<T> power, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Pow(this, x, power, destination);

    public void Copy(ReadOnlySpan<Octonion<T>> source, Span<Octonion<T>> destination)
        => source.CopyTo(destination);

    public void Floor(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Floor(this, x, destination);

    public void Ceiling(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Ceiling(this, x, destination);

    public void Frac(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Frac(this, x, destination);

    public void Sin(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Sin(this, x, destination);

    public void Cos(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Cos(this, x, destination);

    public void MultiplyAdd(ReadOnlySpan<Octonion<T>> x, ReadOnlySpan<Octonion<T>> y, Octonion<T> scalar, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.MultiplyAdd(this, x, y, scalar, destination);

    public void ToFloatSpan(ReadOnlySpan<Octonion<T>> source, Span<float> destination)
        => VectorizedOperationsFallback.ToFloatSpan(this, source, destination);

    public void FromFloatSpan(ReadOnlySpan<float> source, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.FromFloatSpan(this, source, destination);

    public void ToHalfSpan(ReadOnlySpan<Octonion<T>> source, Span<Half> destination)
        => VectorizedOperationsFallback.ToHalfSpan(this, source, destination);

    public void FromHalfSpan(ReadOnlySpan<Half> source, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.FromHalfSpan(this, source, destination);

    public void LeakyReLU(ReadOnlySpan<Octonion<T>> x, Octonion<T> alpha, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.LeakyReLU(this, x, alpha, destination);

    public void GELU(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.GELU(this, x, destination);

    public void Mish(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Mish(this, x, destination);

    public void Swish(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.Swish(this, x, destination);

    public void ELU(ReadOnlySpan<Octonion<T>> x, Octonion<T> alpha, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.ELU(this, x, alpha, destination);

    public void ReLU(ReadOnlySpan<Octonion<T>> x, Span<Octonion<T>> destination)
        => VectorizedOperationsFallback.ReLU(this, x, destination);

    #endregion
}
