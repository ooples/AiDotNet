using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Provides numeric operations for Multivector{T} within a fixed Clifford algebra.
/// </summary>
public class MultivectorOperations<T> : INumericOperations<Multivector<T>>
{
    private readonly INumericOperations<T> _ops;
    private readonly CliffordAlgebra _algebra;

    public MultivectorOperations()
        : this(CliffordAlgebra.Default)
    {
    }

    public MultivectorOperations(CliffordAlgebra algebra)
    {
        _algebra = algebra ?? throw new ArgumentNullException(nameof(algebra));
        _ops = MathHelper.GetNumericOperations<T>();
    }

    public Multivector<T> Add(Multivector<T> a, Multivector<T> b)
    {
        EnsureCompatible(a);
        EnsureCompatible(b);
        return a + b;
    }

    public Multivector<T> Subtract(Multivector<T> a, Multivector<T> b)
    {
        EnsureCompatible(a);
        EnsureCompatible(b);
        return a - b;
    }

    public Multivector<T> Multiply(Multivector<T> a, Multivector<T> b)
    {
        EnsureCompatible(a);
        EnsureCompatible(b);
        return a * b;
    }

    public Multivector<T> Divide(Multivector<T> a, Multivector<T> b)
    {
        EnsureCompatible(a);
        EnsureCompatible(b);
        return a / b;
    }

    public Multivector<T> Negate(Multivector<T> a)
    {
        EnsureCompatible(a);
        return a.Negate();
    }

    public Multivector<T> Zero => new Multivector<T>(_algebra);

    public Multivector<T> One => Multivector<T>.CreateScalar(_algebra, _ops.One);

    public Multivector<T> Sqrt(Multivector<T> value)
    {
        EnsureCompatible(value);
        if (!value.IsScalar)
            throw new NotSupportedException("Sqrt is only defined for scalar multivectors.");

        T scalar = _ops.Sqrt(value.Scalar);
        return Multivector<T>.CreateScalar(_algebra, scalar);
    }

    public Multivector<T> FromDouble(double value)
        => Multivector<T>.CreateScalar(_algebra, _ops.FromDouble(value));

    public int ToInt32(Multivector<T> value)
    {
        EnsureCompatible(value);
        double magnitude = Convert.ToDouble(value.Magnitude);
        return (int)Math.Round(magnitude);
    }

    public bool GreaterThan(Multivector<T> a, Multivector<T> b)
    {
        EnsureCompatible(a);
        EnsureCompatible(b);
        return _ops.GreaterThan(a.Magnitude, b.Magnitude);
    }

    public bool LessThan(Multivector<T> a, Multivector<T> b)
    {
        EnsureCompatible(a);
        EnsureCompatible(b);
        return _ops.LessThan(a.Magnitude, b.Magnitude);
    }

    public Multivector<T> Abs(Multivector<T> value)
    {
        EnsureCompatible(value);
        return Multivector<T>.CreateScalar(_algebra, value.Magnitude);
    }

    public Multivector<T> Square(Multivector<T> value)
    {
        EnsureCompatible(value);
        return value * value;
    }

    public Multivector<T> Exp(Multivector<T> value)
    {
        EnsureCompatible(value);
        if (!value.IsScalar)
            throw new NotSupportedException("Exp is only defined for scalar multivectors.");

        return Multivector<T>.CreateScalar(_algebra, _ops.Exp(value.Scalar));
    }

    public bool Equals(Multivector<T> a, Multivector<T> b)
    {
        if (!a.Algebra.Equals(b.Algebra))
            return false;
        return a.Equals(b);
    }

    public Multivector<T> Power(Multivector<T> baseValue, Multivector<T> exponent)
    {
        EnsureCompatible(baseValue);
        EnsureCompatible(exponent);
        if (!baseValue.IsScalar || !exponent.IsScalar)
            throw new NotSupportedException("Power is only defined for scalar multivectors.");

        return Multivector<T>.CreateScalar(_algebra, _ops.Power(baseValue.Scalar, exponent.Scalar));
    }

    public Multivector<T> Log(Multivector<T> value)
    {
        EnsureCompatible(value);
        if (!value.IsScalar)
            throw new NotSupportedException("Log is only defined for scalar multivectors.");

        return Multivector<T>.CreateScalar(_algebra, _ops.Log(value.Scalar));
    }

    public bool GreaterThanOrEquals(Multivector<T> a, Multivector<T> b)
    {
        EnsureCompatible(a);
        EnsureCompatible(b);
        return _ops.GreaterThanOrEquals(a.Magnitude, b.Magnitude);
    }

    public bool LessThanOrEquals(Multivector<T> a, Multivector<T> b)
    {
        EnsureCompatible(a);
        EnsureCompatible(b);
        return _ops.LessThanOrEquals(a.Magnitude, b.Magnitude);
    }

    public int Compare(Multivector<T> a, Multivector<T> b)
    {
        EnsureCompatible(a);
        EnsureCompatible(b);
        return _ops.Compare(a.Magnitude, b.Magnitude);
    }

    public Multivector<T> Round(Multivector<T> value)
    {
        EnsureCompatible(value);
        var coeffs = new T[value.BasisCount];
        for (int i = 0; i < coeffs.Length; i++)
            coeffs[i] = _ops.Round(value[i]);
        return new Multivector<T>(_algebra, coeffs);
    }

    public Multivector<T> Floor(Multivector<T> value)
    {
        EnsureCompatible(value);
        var coeffs = new T[value.BasisCount];
        for (int i = 0; i < coeffs.Length; i++)
            coeffs[i] = _ops.Floor(value[i]);
        return new Multivector<T>(_algebra, coeffs);
    }

    public Multivector<T> Ceiling(Multivector<T> value)
    {
        EnsureCompatible(value);
        var coeffs = new T[value.BasisCount];
        for (int i = 0; i < coeffs.Length; i++)
            coeffs[i] = _ops.Ceiling(value[i]);
        return new Multivector<T>(_algebra, coeffs);
    }

    public Multivector<T> Frac(Multivector<T> value)
    {
        EnsureCompatible(value);
        var coeffs = new T[value.BasisCount];
        for (int i = 0; i < coeffs.Length; i++)
            coeffs[i] = _ops.Frac(value[i]);
        return new Multivector<T>(_algebra, coeffs);
    }

    public Multivector<T> Sin(Multivector<T> value)
    {
        EnsureCompatible(value);
        var coeffs = new T[value.BasisCount];
        for (int i = 0; i < coeffs.Length; i++)
            coeffs[i] = _ops.Sin(value[i]);
        return new Multivector<T>(_algebra, coeffs);
    }

    public Multivector<T> Cos(Multivector<T> value)
    {
        EnsureCompatible(value);
        var coeffs = new T[value.BasisCount];
        for (int i = 0; i < coeffs.Length; i++)
            coeffs[i] = _ops.Cos(value[i]);
        return new Multivector<T>(_algebra, coeffs);
    }

    public Multivector<T> MinValue
    {
        get
        {
            var coeffs = new T[_algebra.BasisCount];
            for (int i = 0; i < coeffs.Length; i++) { coeffs[i] = _ops.MinValue; }
            return new Multivector<T>(_algebra, coeffs);
        }
    }

    public Multivector<T> MaxValue
    {
        get
        {
            var coeffs = new T[_algebra.BasisCount];
            for (int i = 0; i < coeffs.Length; i++) { coeffs[i] = _ops.MaxValue; }
            return new Multivector<T>(_algebra, coeffs);
        }
    }

    public bool IsNaN(Multivector<T> value)
    {
        EnsureCompatible(value);
        for (int i = 0; i < value.BasisCount; i++)
        {
            if (_ops.IsNaN(value[i]))
                return true;
        }
        return false;
    }

    public bool IsInfinity(Multivector<T> value)
    {
        EnsureCompatible(value);
        for (int i = 0; i < value.BasisCount; i++)
        {
            if (_ops.IsInfinity(value[i]))
                return true;
        }
        return false;
    }

    public Multivector<T> SignOrZero(Multivector<T> value)
    {
        EnsureCompatible(value);
        T magnitude = value.Magnitude;
        if (_ops.Equals(magnitude, _ops.Zero))
            return Zero;

        T invMagnitude = _ops.Divide(_ops.One, magnitude);
        return value.Scale(invMagnitude);
    }

    public int PrecisionBits => _ops.PrecisionBits;

    public float ToFloat(Multivector<T> value)
    {
        EnsureCompatible(value);
        if (!value.IsScalar)
        {
            throw new NotSupportedException(
                "Cannot convert Multivector<T> with non-zero blades to scalar float. " +
                "Extract Scalar explicitly if this is intentional.");
        }
        return _ops.ToFloat(value.Scalar);
    }

    public Multivector<T> FromFloat(float value)
        => Multivector<T>.CreateScalar(_algebra, _ops.FromFloat(value));

    public Half ToHalf(Multivector<T> value)
    {
        EnsureCompatible(value);
        if (!value.IsScalar)
        {
            throw new NotSupportedException(
                "Cannot convert Multivector<T> with non-zero blades to scalar Half. " +
                "Extract Scalar explicitly if this is intentional.");
        }
        return _ops.ToHalf(value.Scalar);
    }

    public Multivector<T> FromHalf(Half value)
        => Multivector<T>.CreateScalar(_algebra, _ops.FromHalf(value));

    public double ToDouble(Multivector<T> value)
    {
        EnsureCompatible(value);
        if (!value.IsScalar)
        {
            throw new NotSupportedException(
                "Cannot convert Multivector<T> with non-zero blades to scalar double. " +
                "Extract Scalar explicitly if this is intentional.");
        }
        return _ops.ToDouble(value.Scalar);
    }

    public bool SupportsCpuAcceleration => false;

    public bool SupportsGpuAcceleration => false;

    #region IVectorizedOperations<Multivector<T>> fallback implementations

    public void Add(ReadOnlySpan<Multivector<T>> x, ReadOnlySpan<Multivector<T>> y, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Add(this, x, y, destination);

    public void Subtract(ReadOnlySpan<Multivector<T>> x, ReadOnlySpan<Multivector<T>> y, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Subtract(this, x, y, destination);

    public void Multiply(ReadOnlySpan<Multivector<T>> x, ReadOnlySpan<Multivector<T>> y, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Multiply(this, x, y, destination);

    public void Divide(ReadOnlySpan<Multivector<T>> x, ReadOnlySpan<Multivector<T>> y, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Divide(this, x, y, destination);

    public Multivector<T> Dot(ReadOnlySpan<Multivector<T>> x, ReadOnlySpan<Multivector<T>> y)
        => VectorizedOperationsFallback.Dot(this, x, y);

    public Multivector<T> Sum(ReadOnlySpan<Multivector<T>> x)
        => VectorizedOperationsFallback.Sum(this, x);

    public Multivector<T> Max(ReadOnlySpan<Multivector<T>> x)
        => VectorizedOperationsFallback.Max(this, x);

    public Multivector<T> Min(ReadOnlySpan<Multivector<T>> x)
        => VectorizedOperationsFallback.Min(this, x);

    public void Exp(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Exp(this, x, destination);

    public void Log(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Log(this, x, destination);

    public void Tanh(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Tanh(this, x, destination);

    public void Sigmoid(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Sigmoid(this, x, destination);

    public void Log2(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Log2(this, x, destination);

    public void SoftMax(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.SoftMax(this, x, destination);

    public Multivector<T> CosineSimilarity(ReadOnlySpan<Multivector<T>> x, ReadOnlySpan<Multivector<T>> y)
        => VectorizedOperationsFallback.CosineSimilarity(this, x, y);

    public void Fill(Span<Multivector<T>> destination, Multivector<T> value) => destination.Fill(value);

    public void MultiplyScalar(ReadOnlySpan<Multivector<T>> x, Multivector<T> scalar, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.MultiplyScalar(this, x, scalar, destination);

    public void DivideScalar(ReadOnlySpan<Multivector<T>> x, Multivector<T> scalar, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.DivideScalar(this, x, scalar, destination);

    public void AddScalar(ReadOnlySpan<Multivector<T>> x, Multivector<T> scalar, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.AddScalar(this, x, scalar, destination);

    public void SubtractScalar(ReadOnlySpan<Multivector<T>> x, Multivector<T> scalar, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.SubtractScalar(this, x, scalar, destination);

    public void Sqrt(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Sqrt(this, x, destination);

    public void Abs(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Abs(this, x, destination);

    public void Negate(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Negate(this, x, destination);

    public void Clip(ReadOnlySpan<Multivector<T>> x, Multivector<T> min, Multivector<T> max, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Clip(this, x, min, max, destination);

    public void Pow(ReadOnlySpan<Multivector<T>> x, Multivector<T> power, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Pow(this, x, power, destination);

    public void Copy(ReadOnlySpan<Multivector<T>> source, Span<Multivector<T>> destination)
        => source.CopyTo(destination);

    public void Floor(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Floor(this, x, destination);

    public void Ceiling(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Ceiling(this, x, destination);

    public void Frac(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Frac(this, x, destination);

    public void Sin(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Sin(this, x, destination);

    public void Cos(ReadOnlySpan<Multivector<T>> x, Span<Multivector<T>> destination)
        => VectorizedOperationsFallback.Cos(this, x, destination);

    #endregion

    private void EnsureCompatible(Multivector<T> value)
    {
        if (!value.Algebra.Equals(_algebra))
            throw new ArgumentException("Multivector algebra does not match this operations instance.");
    }
}
