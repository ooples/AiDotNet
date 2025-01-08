namespace AiDotNet.Wavelets;

public class MorletWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _omega0;

    public MorletWavelet(T? omega0 = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _omega0 = omega0 ?? _numOps.FromDouble(5.0);
    }

    public T Calculate(T x)
    {
        T x2 = _numOps.Square(x);
        T cosine = MathHelper.Cos(_numOps.Multiply(_omega0, x));
        T gaussian = _numOps.Exp(_numOps.Divide(_numOps.Negate(x2), _numOps.FromDouble(2)));

        return _numOps.Multiply(cosine, gaussian);
    }
}