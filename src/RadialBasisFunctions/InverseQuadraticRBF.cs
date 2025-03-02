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

    public T ComputeDerivative(T r)
    {
        // Derivative with respect to r: -2ε²r/(1 + (εr)²)²
        
        // Calculate εr
        T epsilonR = _numOps.Multiply(_epsilon, r);
        
        // Calculate (εr)²
        T epsilonRSquared = _numOps.Multiply(epsilonR, epsilonR);
        
        // Calculate 1 + (εr)²
        T denominator = _numOps.Add(_numOps.One, epsilonRSquared);
        
        // Calculate (1 + (εr)²)²
        T denominatorSquared = _numOps.Multiply(denominator, denominator);
        
        // Calculate ε²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);
        
        // Calculate 2ε²r
        T twoEpsilonSquaredR = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(2.0), epsilonSquared),
            r
        );
        
        // Calculate -2ε²r
        T negativeTwoEpsilonSquaredR = _numOps.Negate(twoEpsilonSquaredR);
        
        // Return -2ε²r/(1 + (εr)²)²
        return _numOps.Divide(negativeTwoEpsilonSquaredR, denominatorSquared);
    }

    public T ComputeWidthDerivative(T r)
    {
        // Derivative with respect to ε: -2εr²/(1 + (εr)²)²
        
        // Calculate εr
        T epsilonR = _numOps.Multiply(_epsilon, r);
        
        // Calculate (εr)²
        T epsilonRSquared = _numOps.Multiply(epsilonR, epsilonR);
        
        // Calculate 1 + (εr)²
        T denominator = _numOps.Add(_numOps.One, epsilonRSquared);
        
        // Calculate (1 + (εr)²)²
        T denominatorSquared = _numOps.Multiply(denominator, denominator);
        
        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);
        
        // Calculate 2εr²
        T twoEpsilonRSquared = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(2.0), _epsilon),
            rSquared
        );
        
        // Calculate -2εr²
        T negativeTwoEpsilonRSquared = _numOps.Negate(twoEpsilonRSquared);
        
        // Return -2εr²/(1 + (εr)²)²
        return _numOps.Divide(negativeTwoEpsilonRSquared, denominatorSquared);
    }
}