namespace AiDotNet.Wavelets;

public class HaarWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public HaarWavelet()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(T x)
    {
        if (_numOps.GreaterThanOrEquals(x, _numOps.Zero) && _numOps.LessThan(x, _numOps.FromDouble(0.5)))
            return _numOps.One;
        if (_numOps.GreaterThanOrEquals(x, _numOps.FromDouble(0.5)) && _numOps.LessThan(x, _numOps.One))
            return _numOps.FromDouble(-1);

        return _numOps.Zero;
    }
}