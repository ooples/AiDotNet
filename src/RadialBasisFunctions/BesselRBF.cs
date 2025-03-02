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

    public T ComputeDerivative(T r)
    {
        // Handle the case when r is very close to zero
        if (MathHelper.AlmostEqual(r, _numOps.Zero))
        {
            // For r=0, the derivative is 0 for nu > 1
            // For nu = 0, the derivative is -epsilon^2/2
            // For nu = 1, the derivative is 0
            if (MathHelper.AlmostEqual(_nu, _numOps.Zero))
            {
                T halfEpsilonSquared = _numOps.Multiply(_numOps.Multiply(_epsilon, _epsilon), _numOps.FromDouble(0.5));
                return _numOps.Negate(halfEpsilonSquared);
            }
            return _numOps.Zero;
        }

        T epsilonR = _numOps.Multiply(_epsilon, r);
    
        // The derivative of J_nu(epsilon*r)/(epsilon*r)^nu with respect to r is:
        // epsilon * [J_(nu-1)(epsilon*r) - (nu/r)*J_nu(epsilon*r)] / (epsilon*r)^nu
    
        T besselNu = MathHelper.BesselJ(_nu, epsilonR);
        T besselNuMinus1 = MathHelper.BesselJ(_numOps.Subtract(_nu, _numOps.One), epsilonR);
    
        T term1 = besselNuMinus1;
        T term2 = _numOps.Multiply(_numOps.Divide(_nu, r), besselNu);
        T numerator = _numOps.Multiply(_epsilon, _numOps.Subtract(term1, term2));
    
        T denominator = _numOps.Power(epsilonR, _nu);
    
        return _numOps.Divide(numerator, denominator);
    }

    public T ComputeWidthDerivative(T r)
    {
        // Handle the case when r is very close to zero
        if (MathHelper.AlmostEqual(r, _numOps.Zero))
        {
            // For r=0, the width derivative depends on nu
            if (MathHelper.AlmostEqual(_nu, _numOps.Zero))
            {
                return _numOps.Zero; // For nu=0, the width derivative at r=0 is 0
            }
            else if (MathHelper.AlmostEqual(_nu, _numOps.One))
            {
                return _numOps.FromDouble(0.5); // For nu=1, the width derivative at r=0 is 0.5
            }
            return _numOps.Zero;
        }

        T epsilonR = _numOps.Multiply(_epsilon, r);
    
        // The derivative of J_nu(epsilon*r)/(epsilon*r)^nu with respect to epsilon is:
        // r * [J_(nu-1)(epsilon*r) - (nu/epsilon)*J_nu(epsilon*r)] / (epsilon*r)^nu - (nu/epsilon) * J_nu(epsilon*r)/(epsilon*r)^nu
        // = r * J_(nu-1)(epsilon*r)/(epsilon*r)^nu - (nu*r/epsilon) * J_nu(epsilon*r)/(epsilon*r)^nu - (nu/epsilon) * J_nu(epsilon*r)/(epsilon*r)^nu
        // = r * J_(nu-1)(epsilon*r)/(epsilon*r)^nu - (nu*(r+1)/epsilon) * J_nu(epsilon*r)/(epsilon*r)^nu
    
        T besselNu = MathHelper.BesselJ(_nu, epsilonR);
        T besselNuMinus1 = MathHelper.BesselJ(_numOps.Subtract(_nu, _numOps.One), epsilonR);
    
        T term1 = _numOps.Multiply(r, besselNuMinus1);
    
        T rPlusOne = _numOps.Add(r, _numOps.One);
        T nuRPlusOne = _numOps.Multiply(_nu, rPlusOne);
        T term2 = _numOps.Multiply(_numOps.Divide(nuRPlusOne, _epsilon), besselNu);
    
        T numerator = _numOps.Subtract(term1, term2);
        T denominator = _numOps.Power(epsilonR, _nu);
    
        return _numOps.Divide(numerator, denominator);
    }
}