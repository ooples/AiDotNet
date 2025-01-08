namespace AiDotNet.Wavelets;

public class GaussianWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public GaussianWavelet()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(T x)
    {
        T x2 = _numOps.Square(x);
        return _numOps.Exp(_numOps.Negate(x2));
    }
}