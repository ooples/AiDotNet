namespace AiDotNet.Wavelets;

public class BattleLemarieWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _order;

    public BattleLemarieWavelet(int order = 1)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
    }

    public T Calculate(T x)
    {
        T result = _numOps.Zero;
        for (int k = -_order; k <= _order; k++)
        {
            T term = BSpline(_numOps.Add(x, _numOps.FromDouble(k)));
            result = _numOps.Add(result, _numOps.Multiply(_numOps.FromDouble(Math.Pow(-1, k)), term));
        }

        return result;
    }

    private T BSpline(T x)
    {
        T absX = _numOps.Abs(x);
        if (_numOps.LessThan(absX, _numOps.FromDouble(0.5)))
        {
            return _numOps.One;
        }
        else if (_numOps.LessThanOrEquals(absX, _numOps.FromDouble(1.5)))
        {
            T temp = _numOps.Subtract(_numOps.FromDouble(1.5), absX);
            return _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Multiply(temp, temp));
        }

        return _numOps.Zero;
    }
}