namespace AiDotNet.Wavelets;

public class ShannonWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public ShannonWavelet()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(T x)
    {
        if (_numOps.Equals(x, _numOps.Zero))
            return _numOps.One;

        T sinc = _numOps.Divide(MathHelper.Sin(x), x);
        T cos = MathHelper.Cos(_numOps.Divide(x, _numOps.FromDouble(2)));

        return _numOps.Multiply(sinc, cos);
    }
}