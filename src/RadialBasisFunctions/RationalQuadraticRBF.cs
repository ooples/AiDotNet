namespace AiDotNet.RadialBasisFunctions;

public class RationalQuadraticRBF<T> : IRadialBasisFunction<T>
{
    private readonly T _epsilon;
    private readonly INumericOperations<T> _numOps;

    public RationalQuadraticRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    public T Compute(T r)
    {
        T rSquared = _numOps.Power(r, _numOps.FromDouble(2));
        T epsilonSquared = _numOps.Power(_epsilon, _numOps.FromDouble(2));
        T denominator = _numOps.Add(rSquared, epsilonSquared);
        T fraction = _numOps.Divide(rSquared, denominator);

        return _numOps.Subtract(_numOps.One, fraction);
    }
}