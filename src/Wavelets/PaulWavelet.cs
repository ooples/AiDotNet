namespace AiDotNet.Wavelets;

public class PaulWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _order;

    public PaulWavelet(int order = 1)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
    }

    public T Calculate(T x)
    {
        Complex<T> i = new Complex<T>(_numOps.Zero, _numOps.One);
        Complex<T> xComplex = new Complex<T>(x, _numOps.Zero);
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();
        
        // Calculate (2^m * i^m * m!) / sqrt(π * (2m)!)
        double m = _order;
        double numerator = Math.Pow(2, m) * Convert.ToDouble(MathHelper.Factorial<T>(_order));
        double denominator = Math.Sqrt(Math.PI * Convert.ToDouble(MathHelper.Factorial<T>(2 * _order)));
        // Calculate (2^m * i^m * m!) / sqrt(π * (2m)!)
        Complex<T> factor = complexOps.Power(i, complexOps.FromDouble(_order)) * complexOps.FromDouble(numerator / denominator);

        // Calculate (1 - ix)^(-m-1)
        Complex<T> base_term = new Complex<T>(_numOps.One, _numOps.Zero) - (i * xComplex);
        T exponentT = _numOps.FromDouble(-m - 1);
        Complex<T> denominator_term = complexOps.Power(base_term, new Complex<T>(exponentT, exponentT));
        Complex<T> result = factor * denominator_term;
        
        return result.ToRealOrComplex<T>();
    }
}