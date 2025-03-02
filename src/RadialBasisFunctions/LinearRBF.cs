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

    public T ComputeDerivative(T r)
    {
        // The derivative of r with respect to r is 1
        return _numOps.One;
    }

    public T ComputeWidthDerivative(T r)
    {
        // Since there is no width parameter, the derivative is 0
        return _numOps.Zero;
    }
}