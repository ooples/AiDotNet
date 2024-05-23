namespace AiDotNet.LinearAlgebra;

public readonly struct Complex
{
    public double Real { get; }
    public double Imaginary { get; }

    public Complex(double real, double imaginary)
    {
        Real = real;
        Imaginary = imaginary;
    }

    public double Magnitude => Math.Sqrt(Real * Real + Imaginary * Imaginary);

    public static Complex operator +(Complex a, Complex b)
    {
        return new Complex(a.Real + b.Real, a.Imaginary + b.Imaginary);
    }

    public static bool operator !=(Complex a, Complex b)
    {
        return a.Real != b.Real && a.Imaginary != b.Imaginary;
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
        return new Complex(a.Real * -1, a.Imaginary * -1);
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

    public static Complex Conjugate(this Complex complex)
    {
        return new Complex(complex.Real, -complex.Imaginary);
    }

    public override string ToString()
    {
        return $"({Real} + {Imaginary}i)";
    }
}