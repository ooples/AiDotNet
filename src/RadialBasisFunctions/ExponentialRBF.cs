namespace AiDotNet.RadialBasisFunctions;

public class ExponentialRBF<T> : IRadialBasisFunction<T>
{
    private readonly T _epsilon;
    private readonly INumericOperations<T> _numOps;

    public ExponentialRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    public T Compute(T r)
    {
        T negativeEpsilonR = _numOps.Multiply(_numOps.Negate(_epsilon), r);
        return _numOps.Exp(negativeEpsilonR);
    }

    public T ComputeDerivative(T r)
    {
        // Derivative with respect to r: -ε * exp(-ε*r)
        T negativeEpsilon = _numOps.Negate(_epsilon);
        return _numOps.Multiply(negativeEpsilon, Compute(r));
    }

    public T ComputeWidthDerivative(T r)
    {
        // Derivative with respect to ε: -r * exp(-ε*r)
        T negativeR = _numOps.Negate(r);
        return _numOps.Multiply(negativeR, Compute(r));
    }
}