namespace AiDotNet.RadialBasisFunctions;

public class SphericalRBF<T> : IRadialBasisFunction<T>
{
    private readonly T _epsilon;
    private readonly INumericOperations<T> _numOps;

    public SphericalRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    public T Compute(T r)
    {
        if (_numOps.GreaterThan(r, _epsilon))
        {
            return _numOps.Zero;
        }

        T rDividedByEpsilon = _numOps.Divide(r, _epsilon);
        T rCubedDividedByEpsilonCubed = _numOps.Power(rDividedByEpsilon, _numOps.FromDouble(3));

        T term1 = _numOps.Multiply(_numOps.FromDouble(1.5), rDividedByEpsilon);
        T term2 = _numOps.Multiply(_numOps.FromDouble(0.5), rCubedDividedByEpsilonCubed);

        return _numOps.Subtract(_numOps.One, _numOps.Subtract(term1, term2));
    }
}