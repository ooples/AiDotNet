using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Represents an octonion (a + b e1 + ... + h e7) with a scalar and seven imaginary components.
/// </summary>
/// <typeparam name="T">The numeric type used for components.</typeparam>
public readonly struct Octonion<T>
{
    private readonly INumericOperations<T> _ops;
    private INumericOperations<T> Ops => _ops ?? MathHelper.GetNumericOperations<T>();

    public T Scalar { get; }
    public T E1 { get; }
    public T E2 { get; }
    public T E3 { get; }
    public T E4 { get; }
    public T E5 { get; }
    public T E6 { get; }
    public T E7 { get; }

    public Octonion(T scalar, T e1, T e2, T e3, T e4, T e5, T e6, T e7)
    {
        Scalar = scalar;
        E1 = e1;
        E2 = e2;
        E3 = e3;
        E4 = e4;
        E5 = e5;
        E6 = e6;
        E7 = e7;
        _ops = MathHelper.GetNumericOperations<T>();
    }

    public T NormSquared
    {
        get
        {
            var ops = Ops;
            T sum = ops.Add(ops.Square(Scalar), ops.Square(E1));
            sum = ops.Add(sum, ops.Square(E2));
            sum = ops.Add(sum, ops.Square(E3));
            sum = ops.Add(sum, ops.Square(E4));
            sum = ops.Add(sum, ops.Square(E5));
            sum = ops.Add(sum, ops.Square(E6));
            sum = ops.Add(sum, ops.Square(E7));
            return sum;
        }
    }

    public T Magnitude => Ops.Sqrt(NormSquared);

    public T VectorMagnitude
    {
        get
        {
            var ops = Ops;
            T sum = ops.Add(ops.Square(E1), ops.Square(E2));
            sum = ops.Add(sum, ops.Square(E3));
            sum = ops.Add(sum, ops.Square(E4));
            sum = ops.Add(sum, ops.Square(E5));
            sum = ops.Add(sum, ops.Square(E6));
            sum = ops.Add(sum, ops.Square(E7));
            return ops.Sqrt(sum);
        }
    }

    public bool IsScalar
    {
        get
        {
            var ops = Ops;
            return ops.Equals(E1, ops.Zero) &&
                   ops.Equals(E2, ops.Zero) &&
                   ops.Equals(E3, ops.Zero) &&
                   ops.Equals(E4, ops.Zero) &&
                   ops.Equals(E5, ops.Zero) &&
                   ops.Equals(E6, ops.Zero) &&
                   ops.Equals(E7, ops.Zero);
        }
    }

    public Octonion<T> Conjugate()
    {
        var ops = Ops;
        return new Octonion<T>(
            Scalar,
            ops.Negate(E1),
            ops.Negate(E2),
            ops.Negate(E3),
            ops.Negate(E4),
            ops.Negate(E5),
            ops.Negate(E6),
            ops.Negate(E7));
    }

    public Octonion<T> Scale(T scalar)
    {
        var ops = Ops;
        return new Octonion<T>(
            ops.Multiply(Scalar, scalar),
            ops.Multiply(E1, scalar),
            ops.Multiply(E2, scalar),
            ops.Multiply(E3, scalar),
            ops.Multiply(E4, scalar),
            ops.Multiply(E5, scalar),
            ops.Multiply(E6, scalar),
            ops.Multiply(E7, scalar));
    }

    public Octonion<T> Inverse()
    {
        var ops = Ops;
        T norm = NormSquared;
        if (ops.Equals(norm, ops.Zero))
            throw new DivideByZeroException("Cannot invert an octonion with zero norm.");

        var conjugate = Conjugate();
        T invNorm = ops.Divide(ops.One, norm);
        return conjugate.Scale(invNorm);
    }

    public static Octonion<T> operator +(Octonion<T> a, Octonion<T> b)
    {
        var ops = a.Ops;
        return new Octonion<T>(
            ops.Add(a.Scalar, b.Scalar),
            ops.Add(a.E1, b.E1),
            ops.Add(a.E2, b.E2),
            ops.Add(a.E3, b.E3),
            ops.Add(a.E4, b.E4),
            ops.Add(a.E5, b.E5),
            ops.Add(a.E6, b.E6),
            ops.Add(a.E7, b.E7));
    }

    public static Octonion<T> operator -(Octonion<T> a, Octonion<T> b)
    {
        var ops = a.Ops;
        return new Octonion<T>(
            ops.Subtract(a.Scalar, b.Scalar),
            ops.Subtract(a.E1, b.E1),
            ops.Subtract(a.E2, b.E2),
            ops.Subtract(a.E3, b.E3),
            ops.Subtract(a.E4, b.E4),
            ops.Subtract(a.E5, b.E5),
            ops.Subtract(a.E6, b.E6),
            ops.Subtract(a.E7, b.E7));
    }

    public static Octonion<T> operator *(Octonion<T> a, Octonion<T> b)
    {
        var ops = a.Ops;
        T a0 = a.Scalar;
        T a1 = a.E1;
        T a2 = a.E2;
        T a3 = a.E3;
        T a4 = a.E4;
        T a5 = a.E5;
        T a6 = a.E6;
        T a7 = a.E7;

        T b0 = b.Scalar;
        T b1 = b.E1;
        T b2 = b.E2;
        T b3 = b.E3;
        T b4 = b.E4;
        T b5 = b.E5;
        T b6 = b.E6;
        T b7 = b.E7;

        T c0 = ops.Multiply(a0, b0);
        c0 = ops.Subtract(c0, ops.Multiply(a1, b1));
        c0 = ops.Subtract(c0, ops.Multiply(a2, b2));
        c0 = ops.Subtract(c0, ops.Multiply(a3, b3));
        c0 = ops.Subtract(c0, ops.Multiply(a4, b4));
        c0 = ops.Subtract(c0, ops.Multiply(a5, b5));
        c0 = ops.Subtract(c0, ops.Multiply(a6, b6));
        c0 = ops.Subtract(c0, ops.Multiply(a7, b7));

        T c1 = ops.Multiply(a0, b1);
        c1 = ops.Add(c1, ops.Multiply(a1, b0));
        c1 = ops.Add(c1, ops.Multiply(a2, b3));
        c1 = ops.Subtract(c1, ops.Multiply(a3, b2));
        c1 = ops.Add(c1, ops.Multiply(a4, b5));
        c1 = ops.Subtract(c1, ops.Multiply(a5, b4));
        c1 = ops.Subtract(c1, ops.Multiply(a6, b7));
        c1 = ops.Add(c1, ops.Multiply(a7, b6));

        T c2 = ops.Multiply(a0, b2);
        c2 = ops.Subtract(c2, ops.Multiply(a1, b3));
        c2 = ops.Add(c2, ops.Multiply(a2, b0));
        c2 = ops.Add(c2, ops.Multiply(a3, b1));
        c2 = ops.Add(c2, ops.Multiply(a4, b6));
        c2 = ops.Add(c2, ops.Multiply(a5, b7));
        c2 = ops.Subtract(c2, ops.Multiply(a6, b4));
        c2 = ops.Subtract(c2, ops.Multiply(a7, b5));

        T c3 = ops.Multiply(a0, b3);
        c3 = ops.Add(c3, ops.Multiply(a1, b2));
        c3 = ops.Subtract(c3, ops.Multiply(a2, b1));
        c3 = ops.Add(c3, ops.Multiply(a3, b0));
        c3 = ops.Add(c3, ops.Multiply(a4, b7));
        c3 = ops.Subtract(c3, ops.Multiply(a5, b6));
        c3 = ops.Add(c3, ops.Multiply(a6, b5));
        c3 = ops.Subtract(c3, ops.Multiply(a7, b4));

        T c4 = ops.Multiply(a0, b4);
        c4 = ops.Subtract(c4, ops.Multiply(a1, b5));
        c4 = ops.Subtract(c4, ops.Multiply(a2, b6));
        c4 = ops.Subtract(c4, ops.Multiply(a3, b7));
        c4 = ops.Add(c4, ops.Multiply(a4, b0));
        c4 = ops.Add(c4, ops.Multiply(a5, b1));
        c4 = ops.Add(c4, ops.Multiply(a6, b2));
        c4 = ops.Add(c4, ops.Multiply(a7, b3));

        T c5 = ops.Multiply(a0, b5);
        c5 = ops.Add(c5, ops.Multiply(a1, b4));
        c5 = ops.Subtract(c5, ops.Multiply(a2, b7));
        c5 = ops.Add(c5, ops.Multiply(a3, b6));
        c5 = ops.Subtract(c5, ops.Multiply(a4, b1));
        c5 = ops.Add(c5, ops.Multiply(a5, b0));
        c5 = ops.Subtract(c5, ops.Multiply(a6, b3));
        c5 = ops.Add(c5, ops.Multiply(a7, b2));

        T c6 = ops.Multiply(a0, b6);
        c6 = ops.Add(c6, ops.Multiply(a1, b7));
        c6 = ops.Add(c6, ops.Multiply(a2, b4));
        c6 = ops.Subtract(c6, ops.Multiply(a3, b5));
        c6 = ops.Subtract(c6, ops.Multiply(a4, b2));
        c6 = ops.Add(c6, ops.Multiply(a5, b3));
        c6 = ops.Add(c6, ops.Multiply(a6, b0));
        c6 = ops.Subtract(c6, ops.Multiply(a7, b1));

        T c7 = ops.Multiply(a0, b7);
        c7 = ops.Subtract(c7, ops.Multiply(a1, b6));
        c7 = ops.Add(c7, ops.Multiply(a2, b5));
        c7 = ops.Add(c7, ops.Multiply(a3, b4));
        c7 = ops.Subtract(c7, ops.Multiply(a4, b3));
        c7 = ops.Subtract(c7, ops.Multiply(a5, b2));
        c7 = ops.Add(c7, ops.Multiply(a6, b1));
        c7 = ops.Add(c7, ops.Multiply(a7, b0));

        return new Octonion<T>(c0, c1, c2, c3, c4, c5, c6, c7);
    }

    public static Octonion<T> operator /(Octonion<T> a, Octonion<T> b)
        => a * b.Inverse();

    public static bool operator ==(Octonion<T> left, Octonion<T> right) => left.Equals(right);

    public static bool operator !=(Octonion<T> left, Octonion<T> right) => !left.Equals(right);

    public bool Equals(Octonion<T> other)
    {
        var ops = Ops;
        return ops.Equals(Scalar, other.Scalar) &&
               ops.Equals(E1, other.E1) &&
               ops.Equals(E2, other.E2) &&
               ops.Equals(E3, other.E3) &&
               ops.Equals(E4, other.E4) &&
               ops.Equals(E5, other.E5) &&
               ops.Equals(E6, other.E6) &&
               ops.Equals(E7, other.E7);
    }

    public override bool Equals(object? obj) => obj is Octonion<T> other && Equals(other);

    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 23 + (Scalar?.GetHashCode() ?? 0);
            hash = hash * 23 + (E1?.GetHashCode() ?? 0);
            hash = hash * 23 + (E2?.GetHashCode() ?? 0);
            hash = hash * 23 + (E3?.GetHashCode() ?? 0);
            hash = hash * 23 + (E4?.GetHashCode() ?? 0);
            hash = hash * 23 + (E5?.GetHashCode() ?? 0);
            hash = hash * 23 + (E6?.GetHashCode() ?? 0);
            hash = hash * 23 + (E7?.GetHashCode() ?? 0);
            return hash;
        }
    }

    public override string ToString()
    {
        return $"{Scalar} + {E1}e1 + {E2}e2 + {E3}e3 + {E4}e4 + {E5}e5 + {E6}e6 + {E7}e7";
    }
}
