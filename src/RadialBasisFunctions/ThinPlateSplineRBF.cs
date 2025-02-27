namespace AiDotNet.RadialBasisFunctions;

public class ThinPlateSplineRBF<T> : IRadialBasisFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public ThinPlateSplineRBF()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Compute(T r)
    {
        if (_numOps.Equals(r, _numOps.Zero))
        {
            return _numOps.Zero;
        }
        
        T r2 = _numOps.Multiply(r, r);
        return _numOps.Multiply(r2, _numOps.Log(r));
    }
}