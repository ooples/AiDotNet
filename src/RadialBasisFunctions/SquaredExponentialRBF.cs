namespace AiDotNet.RadialBasisFunctions;

public class SquaredExponentialRBF<T> : IRadialBasisFunction<T>
{
    private readonly T _epsilon;
    private readonly INumericOperations<T> _numOps;

    public SquaredExponentialRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    public T Compute(T r)
    {
        T epsilonR = _numOps.Multiply(_epsilon, r);
        T squaredEpsilonR = _numOps.Power(epsilonR, _numOps.FromDouble(2));
        T negativeSquaredEpsilonR = _numOps.Negate(squaredEpsilonR);

        return _numOps.Exp(negativeSquaredEpsilonR);
    }
}