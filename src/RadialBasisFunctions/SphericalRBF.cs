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

    public T ComputeDerivative(T r)
    {
        // For r > ε, the derivative is 0
        if (_numOps.GreaterThan(r, _epsilon))
        {
            return _numOps.Zero;
        }
        
        // Calculate r/ε
        T rDividedByEpsilon = _numOps.Divide(r, _epsilon);
        
        // Calculate (r/ε)²
        T rDividedByEpsilonSquared = _numOps.Multiply(rDividedByEpsilon, rDividedByEpsilon);
        
        // Calculate (r/ε)² - 1
        T term = _numOps.Subtract(rDividedByEpsilonSquared, _numOps.One);
        
        // Calculate 1.5/ε
        T factor = _numOps.Divide(_numOps.FromDouble(1.5), _epsilon);
        
        // Return (1.5/ε)[(r/ε)² - 1]
        return _numOps.Multiply(factor, term);
    }

    public T ComputeWidthDerivative(T r)
    {
        // For r > ε, the width derivative requires special handling
        if (_numOps.GreaterThan(r, _epsilon))
        {
            // The derivative at the boundary is a delta function, which we can't represent directly
            // For practical purposes, we return 0 for r > ε
            return _numOps.Zero;
        }
        
        // Calculate r/ε
        T rDividedByEpsilon = _numOps.Divide(r, _epsilon);
        
        // Calculate (r/ε)²
        T rDividedByEpsilonSquared = _numOps.Multiply(rDividedByEpsilon, rDividedByEpsilon);
        
        // Calculate 1 - (r/ε)²
        T term = _numOps.Subtract(_numOps.One, rDividedByEpsilonSquared);
        
        // Calculate 1.5r/ε²
        T factor = _numOps.Divide(
            _numOps.Multiply(_numOps.FromDouble(1.5), r),
            _numOps.Multiply(_epsilon, _epsilon)
        );
        
        // Return (1.5r/ε²)[1 - (r/ε)²]
        return _numOps.Multiply(factor, term);
    }
}