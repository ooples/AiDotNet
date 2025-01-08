namespace AiDotNet.Wavelets;

public class ComplexMorletWavelet<T> : IWaveletFunction<Complex<T>>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _omega0;
    private readonly T _sigma;

    public ComplexMorletWavelet(T? omega0 = default, T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _omega0 = omega0 ?? _numOps.FromDouble(5.0);
        _sigma = sigma ?? _numOps.One;
    }

    public Complex<T> Calculate(Complex<T> z)
    {
        T x = z.Real;
        T y = z.Imaginary;

        T expTerm = _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Add(_numOps.Multiply(x, x), _numOps.Multiply(y, y)), _numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(_sigma, _sigma)))));
        T cosTerm = MathHelper.Cos(_numOps.Multiply(_omega0, x));
        T sinTerm = MathHelper.Sin(_numOps.Multiply(_omega0, x));

        T real = _numOps.Multiply(expTerm, cosTerm);
        T imag = _numOps.Multiply(expTerm, sinTerm);

        return new Complex<T>(real, imag);
    }
}