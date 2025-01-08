namespace AiDotNet.Wavelets;

public class MexicanHatWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public MexicanHatWavelet()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(T x)
    {
        T x2 = _numOps.Square(x);
        T exp_term = _numOps.Exp(_numOps.Negate(_numOps.Divide(x2, _numOps.FromDouble(2))));
        
        T term1 = _numOps.Subtract(_numOps.One, x2);
        T result = _numOps.Multiply(term1, exp_term);

        // Normalization factor
        double norm_factor = 2.0 / (Math.Sqrt(3) * Math.Pow(Math.PI, 0.25));
        result = _numOps.Multiply(result, _numOps.FromDouble(norm_factor));

        return result;
    }
}