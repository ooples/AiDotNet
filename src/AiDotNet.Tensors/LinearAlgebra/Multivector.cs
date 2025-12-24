using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Represents a multivector in a Clifford algebra.
/// </summary>
/// <typeparam name="T">The numeric type used for coefficients.</typeparam>
public sealed class Multivector<T> : IEquatable<Multivector<T>>
{
    private readonly INumericOperations<T> _ops;
    private readonly T[] _coefficients;

    public CliffordAlgebra Algebra { get; }

    public int BasisCount => _coefficients.Length;

    public T Scalar => _coefficients[0];

    public Multivector(CliffordAlgebra algebra)
        : this(algebra, MathHelper.GetNumericOperations<T>(), CreateZeroCoefficients(algebra, MathHelper.GetNumericOperations<T>()))
    {
    }

    public Multivector(CliffordAlgebra algebra, IReadOnlyList<T> coefficients)
        : this(algebra, MathHelper.GetNumericOperations<T>(), CopyCoefficients(algebra, coefficients))
    {
    }

    public Multivector(CliffordAlgebra algebra, IReadOnlyDictionary<int, T> coefficients)
        : this(algebra, MathHelper.GetNumericOperations<T>(), CreateSparseCoefficients(algebra, MathHelper.GetNumericOperations<T>(), coefficients))
    {
    }

    private Multivector(CliffordAlgebra algebra, INumericOperations<T> ops, T[] coefficients)
    {
        Algebra = algebra ?? throw new ArgumentNullException(nameof(algebra));
        _ops = ops ?? throw new ArgumentNullException(nameof(ops));
        _coefficients = coefficients ?? throw new ArgumentNullException(nameof(coefficients));
    }

    /// <summary>
    /// Gets the coefficient for the specified blade index.
    /// </summary>
    /// <remarks>
    /// Multivector is immutable. Use factory methods or constructors to create modified instances.
    /// </remarks>
    public T this[int blade]
    {
        get
        {
            ValidateBlade(blade);
            return _coefficients[blade];
        }
    }

    public bool IsScalar
    {
        get
        {
            for (int i = 1; i < _coefficients.Length; i++)
            {
                if (!_ops.Equals(_coefficients[i], _ops.Zero))
                    return false;
            }
            return true;
        }
    }

    public bool IsZero
    {
        get
        {
            for (int i = 0; i < _coefficients.Length; i++)
            {
                if (!_ops.Equals(_coefficients[i], _ops.Zero))
                    return false;
            }
            return true;
        }
    }

    public T Magnitude
    {
        get
        {
            T sum = _ops.Zero;
            for (int i = 0; i < _coefficients.Length; i++)
                sum = _ops.Add(sum, _ops.Square(_coefficients[i]));
            return _ops.Sqrt(sum);
        }
    }

    public Multivector<T> Negate()
    {
        var result = new T[_coefficients.Length];
        for (int i = 0; i < _coefficients.Length; i++)
            result[i] = _ops.Negate(_coefficients[i]);
        return new Multivector<T>(Algebra, _ops, result);
    }

    public Multivector<T> Scale(T scalar)
    {
        var result = new T[_coefficients.Length];
        for (int i = 0; i < _coefficients.Length; i++)
            result[i] = _ops.Multiply(_coefficients[i], scalar);
        return new Multivector<T>(Algebra, _ops, result);
    }

    public Multivector<T> Reverse()
    {
        var result = new T[_coefficients.Length];
        for (int blade = 0; blade < _coefficients.Length; blade++)
        {
            int grade = Algebra.GetGrade(blade);
            int sign = ((grade * (grade - 1) / 2) & 1) == 0 ? 1 : -1;
            result[blade] = sign < 0 ? _ops.Negate(_coefficients[blade]) : _coefficients[blade];
        }
        return new Multivector<T>(Algebra, _ops, result);
    }

    public Multivector<T> GeometricProduct(Multivector<T> other)
    {
        EnsureSameAlgebra(other);

        var result = CreateZeroCoefficients(Algebra, _ops);
        for (int i = 0; i < _coefficients.Length; i++)
        {
            if (_ops.Equals(_coefficients[i], _ops.Zero))
                continue;

            for (int j = 0; j < _coefficients.Length; j++)
            {
                if (_ops.Equals(other._coefficients[j], _ops.Zero))
                    continue;

                if (!Algebra.TryMultiplyBlades(i, j, out int blade, out int sign))
                    continue;

                T term = _ops.Multiply(_coefficients[i], other._coefficients[j]);
                if (sign < 0)
                    term = _ops.Negate(term);

                result[blade] = _ops.Add(result[blade], term);
            }
        }

        return new Multivector<T>(Algebra, _ops, result);
    }

    public Multivector<T> OuterProduct(Multivector<T> other)
    {
        EnsureSameAlgebra(other);

        var result = CreateZeroCoefficients(Algebra, _ops);
        for (int i = 0; i < _coefficients.Length; i++)
        {
            if (_ops.Equals(_coefficients[i], _ops.Zero))
                continue;

            for (int j = 0; j < _coefficients.Length; j++)
            {
                if (_ops.Equals(other._coefficients[j], _ops.Zero))
                    continue;

                if ((i & j) != 0)
                    continue;

                int sign = Algebra.ReorderingSign(i, j);
                int blade = i ^ j;
                T term = _ops.Multiply(_coefficients[i], other._coefficients[j]);
                if (sign < 0)
                    term = _ops.Negate(term);
                result[blade] = _ops.Add(result[blade], term);
            }
        }

        return new Multivector<T>(Algebra, _ops, result);
    }

    public Multivector<T> InnerProduct(Multivector<T> other)
    {
        EnsureSameAlgebra(other);

        var result = CreateZeroCoefficients(Algebra, _ops);
        for (int i = 0; i < _coefficients.Length; i++)
        {
            if (_ops.Equals(_coefficients[i], _ops.Zero))
                continue;

            int gradeA = Algebra.GetGrade(i);
            for (int j = 0; j < _coefficients.Length; j++)
            {
                if (_ops.Equals(other._coefficients[j], _ops.Zero))
                    continue;

                int gradeB = Algebra.GetGrade(j);
                if (gradeA > gradeB)
                    continue;

                if (!Algebra.TryMultiplyBlades(i, j, out int blade, out int sign))
                    continue;

                int gradeResult = Algebra.GetGrade(blade);
                if (gradeResult != gradeB - gradeA)
                    continue;

                T term = _ops.Multiply(_coefficients[i], other._coefficients[j]);
                if (sign < 0)
                    term = _ops.Negate(term);

                result[blade] = _ops.Add(result[blade], term);
            }
        }

        return new Multivector<T>(Algebra, _ops, result);
    }

    public Multivector<T> Inverse()
    {
        var reverse = Reverse();
        var product = GeometricProduct(reverse);
        if (!product.IsScalar)
            throw new NotSupportedException("Inverse is only defined for blades with scalar reverse product.");

        var scalar = product.Scalar;
        if (_ops.Equals(scalar, _ops.Zero))
            throw new DivideByZeroException("Cannot invert a multivector with zero scalar norm.");

        T invScalar = _ops.Divide(_ops.One, scalar);
        return reverse.Scale(invScalar);
    }

    public static Multivector<T> CreateScalar(CliffordAlgebra algebra, T scalar)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var coeffs = CreateZeroCoefficients(algebra, ops);
        coeffs[0] = scalar;
        return new Multivector<T>(algebra, ops, coeffs);
    }

    public static Multivector<T> operator +(Multivector<T> left, Multivector<T> right)
    {
        left.EnsureSameAlgebra(right);
        var result = new T[left._coefficients.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = left._ops.Add(left._coefficients[i], right._coefficients[i]);
        return new Multivector<T>(left.Algebra, left._ops, result);
    }

    public static Multivector<T> operator -(Multivector<T> left, Multivector<T> right)
    {
        left.EnsureSameAlgebra(right);
        var result = new T[left._coefficients.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = left._ops.Subtract(left._coefficients[i], right._coefficients[i]);
        return new Multivector<T>(left.Algebra, left._ops, result);
    }

    public static Multivector<T> operator *(Multivector<T> left, Multivector<T> right)
        => left.GeometricProduct(right);

    public static Multivector<T> operator /(Multivector<T> left, Multivector<T> right)
        => left.GeometricProduct(right.Inverse());

    public bool Equals(Multivector<T>? other)
    {
        if (other is null)
            return false;
        if (!Algebra.Equals(other.Algebra))
            return false;
        if (_coefficients.Length != other._coefficients.Length)
            return false;

        for (int i = 0; i < _coefficients.Length; i++)
        {
            if (!_ops.Equals(_coefficients[i], other._coefficients[i]))
                return false;
        }
        return true;
    }

    public override bool Equals(object? obj) => obj is Multivector<T> other && Equals(other);

    public override int GetHashCode()
    {
        unchecked
        {
            int hash = Algebra.GetHashCode();
            for (int i = 0; i < _coefficients.Length; i++)
                hash = hash * 23 + (_coefficients[i]?.GetHashCode() ?? 0);
            return hash;
        }
    }

    public override string ToString()
    {
        return $"Multivector(dim={Algebra.Dimension}, blades={BasisCount})";
    }

    private void ValidateBlade(int blade)
    {
        if (blade < 0 || blade >= _coefficients.Length)
            throw new ArgumentOutOfRangeException(nameof(blade));
    }

    private void EnsureSameAlgebra(Multivector<T> other)
    {
        if (!Algebra.Equals(other.Algebra))
            throw new ArgumentException("Multivectors must belong to the same algebra.");
    }

    private static T[] CreateZeroCoefficients(CliffordAlgebra algebra, INumericOperations<T> ops)
    {
        if (algebra is null)
            throw new ArgumentNullException(nameof(algebra));
        if (ops is null)
            throw new ArgumentNullException(nameof(ops));

        var coeffs = new T[algebra.BasisCount];
        for (int i = 0; i < coeffs.Length; i++) { coeffs[i] = ops.Zero; }
        return coeffs;
    }

    private static T[] CopyCoefficients(CliffordAlgebra algebra, IReadOnlyList<T> coefficients)
    {
        if (algebra is null)
            throw new ArgumentNullException(nameof(algebra));
        if (coefficients is null)
            throw new ArgumentNullException(nameof(coefficients));
        if (coefficients.Count != algebra.BasisCount)
            throw new ArgumentException("Coefficient count does not match algebra basis count.", nameof(coefficients));

        var ops = MathHelper.GetNumericOperations<T>();
        var result = CreateZeroCoefficients(algebra, ops);
        for (int i = 0; i < coefficients.Count; i++)
            result[i] = coefficients[i];
        return result;
    }

    private static T[] CreateSparseCoefficients(CliffordAlgebra algebra, INumericOperations<T> ops, IReadOnlyDictionary<int, T> coefficients)
    {
        if (coefficients is null)
            throw new ArgumentNullException(nameof(coefficients));

        var result = CreateZeroCoefficients(algebra, ops);
        foreach (var kvp in coefficients)
        {
            if (kvp.Key < 0 || kvp.Key >= result.Length)
                throw new ArgumentOutOfRangeException(nameof(coefficients), "Blade index is out of range.");

            result[kvp.Key] = kvp.Value;
        }
        return result;
    }
}
