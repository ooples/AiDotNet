namespace AiDotNet.Wavelets;

public class GaborWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _omega0;
    private readonly T _sigma;

    public GaborWavelet(T? omega0 = default, T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _omega0 = omega0 ?? _numOps.FromDouble(5.0);
        _sigma = sigma ?? _numOps.One;
    }

    public T Calculate(T x)
    {
        T gaussianTerm = _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Multiply(x, x), _numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(_sigma, _sigma)))));
        T cosTerm = MathHelper.Cos(_numOps.Multiply(_omega0, x));

        return _numOps.Multiply(gaussianTerm, cosTerm);
    }
}