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

    public T ComputeDerivative(T r)
    {
        // Derivative with respect to r: -2ε²r * exp(-(εr)²)
        
        // Calculate εr
        T epsilonR = _numOps.Multiply(_epsilon, r);
        
        // Calculate (εr)²
        T squaredEpsilonR = _numOps.Multiply(epsilonR, epsilonR);
        
        // Calculate -(εr)²
        T negativeSquaredEpsilonR = _numOps.Negate(squaredEpsilonR);
        
        // Calculate exp(-(εr)²)
        T expTerm = _numOps.Exp(negativeSquaredEpsilonR);
        
        // Calculate ε²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);
        
        // Calculate 2ε²r
        T twoEpsilonSquaredR = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(2.0), epsilonSquared),
            r
        );
        
        // Calculate -2ε²r
        T negativeTwoEpsilonSquaredR = _numOps.Negate(twoEpsilonSquaredR);
        
        // Return -2ε²r * exp(-(εr)²)
        return _numOps.Multiply(negativeTwoEpsilonSquaredR, expTerm);
    }

    public T ComputeWidthDerivative(T r)
    {
        // Derivative with respect to ε: -2εr² * exp(-(εr)²)
        
        // Calculate εr
        T epsilonR = _numOps.Multiply(_epsilon, r);
        
        // Calculate (εr)²
        T squaredEpsilonR = _numOps.Multiply(epsilonR, epsilonR);
        
        // Calculate -(εr)²
        T negativeSquaredEpsilonR = _numOps.Negate(squaredEpsilonR);
        
        // Calculate exp(-(εr)²)
        T expTerm = _numOps.Exp(negativeSquaredEpsilonR);
        
        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);
        
        // Calculate 2εr²
        T twoEpsilonRSquared = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(2.0), _epsilon),
            rSquared
        );
        
        // Calculate -2εr²
        T negativeTwoEpsilonRSquared = _numOps.Negate(twoEpsilonRSquared);
        
        // Return -2εr² * exp(-(εr)²)
        return _numOps.Multiply(negativeTwoEpsilonRSquared, expTerm);
    }
}