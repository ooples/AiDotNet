namespace AiDotNet.LinearAlgebra;

public readonly struct Complex(double real, double imaginary) : IEquatable<Complex>
{
    public double Real { get; } = real;
    public double Imaginary { get; } = imaginary;

    public double Magnitude => Math.Sqrt(Real * Real + Imaginary * Imaginary);
    public double Phase => Math.Atan2(Imaginary, Real);

    public static Complex operator +(Complex a, Complex b)
    {
        return new Complex(a.Real + b.Real, a.Imaginary + b.Imaginary);
    }

    public static bool operator !=(Complex a, Complex b)
    {
        return !(a == b);
    }

    public static bool operator ==(Complex a, Complex b)
    {
        return a.Real == b.Real && a.Imaginary == b.Imaginary;
    }

    public static Complex operator -(Complex a, Complex b)
    {
        return new Complex(a.Real - b.Real, a.Imaginary - b.Imaginary);
    }

    public static Complex operator -(Complex a)
    {
        return new Complex(-a.Real, -a.Imaginary);
    }

    public static Complex operator *(Complex a, Complex b)
    {
        return new Complex(a.Real * b.Real - a.Imaginary * b.Imaginary, a.Real * b.Imaginary + a.Imaginary * b.Real);
    }

    public static Complex operator /(Complex a, Complex b)
    {
        double denominator = b.Real * b.Real + b.Imaginary * b.Imaginary;
        return new Complex((a.Real * b.Real + a.Imaginary * b.Imaginary) / denominator, (a.Imaginary * b.Real - a.Real * b.Imaginary) / denominator);
    }

    public override bool Equals(object? obj)
    {
        return obj is Complex complex && Equals(complex);
    }

    public bool Equals(Complex other)
    {
        return Real == other.Real && Imaginary == other.Imaginary;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(Real, Imaginary);
    }

    public Complex Conjugate()
    {
        return new Complex(Real, -Imaginary);
    }

    public override string ToString()
    {
        return $"({Real} + {Imaginary}i)";
    }
}