namespace AiDotNet.RadialBasisFunctions;

public class InverseQuadraticRBF<T> : IRadialBasisFunction<T>
{
    private readonly T _epsilon;
    private readonly INumericOperations<T> _numOps;

    public InverseQuadraticRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    public T Compute(T r)
    {
        T epsilonR = _numOps.Multiply(_epsilon, r);
        T denominator = _numOps.Add(_numOps.One, _numOps.Power(epsilonR, _numOps.FromDouble(2)));

        return _numOps.Divide(_numOps.One, denominator);
    }
}