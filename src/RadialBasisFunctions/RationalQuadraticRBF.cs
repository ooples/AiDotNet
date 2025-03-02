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

    public T ComputeDerivative(T r)
    {
        // Derivative with respect to r: -2rε²/(r² + ε²)²
        
        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);
        
        // Calculate ε²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);
        
        // Calculate r² + ε²
        T denominator = _numOps.Add(rSquared, epsilonSquared);
        
        // Calculate (r² + ε²)²
        T denominatorSquared = _numOps.Multiply(denominator, denominator);
        
        // Calculate 2rε²
        T twoREpsilonSquared = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(2.0), r),
            epsilonSquared
        );
        
        // Calculate -2rε²
        T negativeTwoREpsilonSquared = _numOps.Negate(twoREpsilonSquared);
        
        // Return -2rε²/(r² + ε²)²
        return _numOps.Divide(negativeTwoREpsilonSquared, denominatorSquared);
    }

    public T ComputeWidthDerivative(T r)
    {
        // Derivative with respect to ε: 2εr²/(r² + ε²)²
        
        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);
        
        // Calculate ε²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);
        
        // Calculate r² + ε²
        T denominator = _numOps.Add(rSquared, epsilonSquared);
        
        // Calculate (r² + ε²)²
        T denominatorSquared = _numOps.Multiply(denominator, denominator);
        
        // Calculate 2εr²
        T twoEpsilonRSquared = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(2.0), _epsilon),
            rSquared
        );
        
        // Return 2εr²/(r² + ε²)²
        return _numOps.Divide(twoEpsilonRSquared, denominatorSquared);
    }
}