namespace AiDotNet.RadialBasisFunctions;

public class GaussianRBF<T> : IRadialBasisFunction<T>
{
    private readonly T _epsilon;
    private readonly INumericOperations<T> _numOps;

    public GaussianRBF(double epsilon = 1e-16)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    public T Compute(T r)
    {
        return _numOps.Exp(_numOps.Negate(_numOps.Multiply(_epsilon, _numOps.Multiply(r, r))));
    }
}