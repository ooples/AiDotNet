namespace AiDotNet.RadialBasisFunctions;

public class BesselRBF<T> : IRadialBasisFunction<T>
{
    private readonly T _epsilon;
    private readonly T _nu;
    private readonly INumericOperations<T> _numOps;

    public BesselRBF(double epsilon = 1.0, double nu = 0.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
        _nu = _numOps.FromDouble(nu);
    }

    public T Compute(T r)
    {
        T epsilonR = _numOps.Multiply(_epsilon, r);
        
        // Handle the case when epsilonR is very close to zero
        if (MathHelper.AlmostEqual(epsilonR, _numOps.Zero))
        {
            return _numOps.One;
        }

        T besselValue = MathHelper.BesselJ(_nu, epsilonR);
        T denominator = _numOps.Power(epsilonR, _nu);

        return _numOps.Divide(besselValue, denominator);
    }
}