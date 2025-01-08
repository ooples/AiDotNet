namespace AiDotNet.Wavelets;

public class BSplineWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _order;

    public BSplineWavelet(int order = 3)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
    }

    public T Calculate(T x)
    {
        return BSpline(x, _order);
    }

    private T BSpline(T x, int n)
    {
        if (n == 0)
        {
            return (_numOps.GreaterThanOrEquals(x, _numOps.Zero) && _numOps.LessThan(x, _numOps.One)) ? _numOps.One : _numOps.Zero;
        }

        T term1 = _numOps.Multiply(_numOps.Divide(x, _numOps.FromDouble(n)), BSpline(_numOps.Subtract(x, _numOps.One), n - 1));
        T term2 = _numOps.Multiply(_numOps.Divide(_numOps.Subtract(_numOps.FromDouble(n + 1), x), _numOps.FromDouble(n)), BSpline(x, n - 1));

        return _numOps.Add(term1, term2);
    }
}