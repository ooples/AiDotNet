namespace AiDotNet.RadialBasisFunctions;

public class LinearRBF<T> : IRadialBasisFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public LinearRBF()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Compute(T r)
    {
        return r;
    }
}