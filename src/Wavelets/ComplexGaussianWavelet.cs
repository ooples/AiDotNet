namespace AiDotNet.Wavelets;

public class ComplexGaussianWavelet<T> : IWaveletFunction<Complex<T>>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _order;

    public ComplexGaussianWavelet(int order = 1)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
    }

    public Complex<T> Calculate(Complex<T> z)
    {
        T x = z.Real;
        T y = z.Imaginary;

        T gaussianTerm = _numOps.Exp(_numOps.Negate(_numOps.Multiply(x, x)));
        T polynomialTerm = HermitePolynomial(x, _order);

        T real = _numOps.Multiply(gaussianTerm, polynomialTerm);
        T imag = _numOps.Zero; // The imaginary part is zero for real input

        return new Complex<T>(real, imag);
    }

    private T HermitePolynomial(T x, int n)
    {
        if (n == 0) return _numOps.One;
        if (n == 1) return _numOps.Multiply(_numOps.FromDouble(2), x);

        T h0 = _numOps.One;
        T h1 = _numOps.Multiply(_numOps.FromDouble(2), x);

        for (int i = 2; i <= n; i++)
        {
            T hi = _numOps.Subtract(_numOps.Multiply(_numOps.Multiply(_numOps.FromDouble(2), x), h1), _numOps.Multiply(_numOps.FromDouble(2 * (i - 1)), h0));
            h0 = h1;
            h1 = hi;
        }

        return h1;
    }
}