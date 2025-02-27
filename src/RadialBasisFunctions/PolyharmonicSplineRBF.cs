namespace AiDotNet.RadialBasisFunctions;

public class PolyharmonicSplineRBF<T> : IRadialBasisFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _k;

    public PolyharmonicSplineRBF(int k = 2)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _k = k;
    }

    public T Compute(T r)
    {
        if (_numOps.Equals(r, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        if (_k % 2 == 0)
        {
            // For even k: r^k * log(r)
            T rPowK = _numOps.Power(r, _numOps.FromDouble(_k));
            return _numOps.Multiply(rPowK, _numOps.Log(r));
        }
        else
        {
            // For odd k: r^k
            return _numOps.Power(r, _numOps.FromDouble(_k));
        }
    }
}