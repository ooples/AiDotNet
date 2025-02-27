namespace AiDotNet.RadialBasisFunctions;

public class CubicRBF<T> : IRadialBasisFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public CubicRBF()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Compute(T r)
    {
        return _numOps.Multiply(r, _numOps.Multiply(r, r));
    }
}