namespace AiDotNet.Wavelets;

public class DOGWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _order;

    public DOGWavelet(int order = 2)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
    }

    public T Calculate(T x)
    {
        T x2 = _numOps.Square(x);
        T exp_term = _numOps.Exp(_numOps.Negate(_numOps.Divide(x2, _numOps.FromDouble(2))));
        
        T result = _numOps.FromDouble(Math.Pow(-1, _order));
        for (int i = 0; i < _order; i++)
        {
            result = _numOps.Multiply(result, x);
        }
        result = _numOps.Multiply(result, exp_term);

        // Normalization factor
        double norm_factor = Math.Pow(-1, _order) / (Math.Sqrt(Convert.ToDouble(MathHelper.Factorial<T>(_order))) * Math.Pow(2, (_order + 1.0) / 2.0));
        result = _numOps.Multiply(result, _numOps.FromDouble(norm_factor));

        return result;
    }
}