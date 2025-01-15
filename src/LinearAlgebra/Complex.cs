namespace AiDotNet.LinearAlgebra;
public readonly struct Complex<T>
{
    private readonly INumericOperations<T> ops;

    public T Real { get; }
    public T Imaginary { get; }

    public Complex(T real, T imaginary)
    {
        Real = real;
        Imaginary = imaginary;
        ops = MathHelper.GetNumericOperations<T>();
    }

    public T Magnitude => ops.Sqrt(ops.Add(ops.Square(Real), ops.Square(Imaginary)));

    public T Phase => ops.FromDouble(Math.Atan2(Convert.ToDouble(Imaginary), Convert.ToDouble(Real)));

    public static Complex<T> operator +(Complex<T> a, Complex<T> b)
        => new(a.ops.Add(a.Real, b.Real), a.ops.Add(a.Imaginary, b.Imaginary));

    public static Complex<T> operator -(Complex<T> a, Complex<T> b)
        => new(a.ops.Subtract(a.Real, b.Real), a.ops.Subtract(a.Imaginary, b.Imaginary));

    public static Complex<T> operator *(Complex<T> a, Complex<T> b)
        => new(
            a.ops.Subtract(a.ops.Multiply(a.Real, b.Real), a.ops.Multiply(a.Imaginary, b.Imaginary)),
            a.ops.Add(a.ops.Multiply(a.Real, b.Imaginary), a.ops.Multiply(a.Imaginary, b.Real))
        );

    public static Complex<T> operator /(Complex<T> a, Complex<T> b)
    {
        T denominator = a.ops.Add(a.ops.Square(b.Real), a.ops.Square(b.Imaginary));
        return new Complex<T>(
            a.ops.Divide(a.ops.Add(a.ops.Multiply(a.Real, b.Real), a.ops.Multiply(a.Imaginary, b.Imaginary)), denominator),
            a.ops.Divide(a.ops.Subtract(a.ops.Multiply(a.Imaginary, b.Real), a.ops.Multiply(a.Real, b.Imaginary)), denominator)
        );
    }

    public static bool operator ==(Complex<T> left, Complex<T> right)
        => left.Equals(right);

    public static bool operator !=(Complex<T> left, Complex<T> right)
        => !left.Equals(right);

    public override bool Equals(object? obj)
        => obj is Complex<T> complex && Equals(complex);

    public bool Equals(Complex<T> other)
        => ops.Equals(Real, other.Real) && ops.Equals(Imaginary, other.Imaginary);

    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 23 + (Real?.GetHashCode() ?? 0);
            hash = hash * 23 + (Imaginary?.GetHashCode() ?? 0);
            return hash;
        }
    }

    public Complex<T> Conjugate()
        => new(Real, ops.Negate(Imaginary));

    public override string ToString()
        => $"{Real} + {Imaginary}i";

    public static Complex<T> FromPolarCoordinates(T magnitude, T phase)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return new Complex<T>(
            ops.Multiply(magnitude, MathHelper.Cos(phase)),
            ops.Multiply(magnitude, MathHelper.Sin(phase))
        );
    }
}