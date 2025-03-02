namespace AiDotNet.RadialBasisFunctions;

public class InverseMultiquadricRBF<T> : IRadialBasisFunction<T>
{
    private readonly T _epsilon;
    private readonly INumericOperations<T> _numOps;

    public InverseMultiquadricRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    public T Compute(T r)
    {
        return _numOps.Divide(
            _numOps.One,
            _numOps.Sqrt(_numOps.Add(_numOps.Multiply(r, r), _numOps.Multiply(_epsilon, _epsilon)))
        );
    }

    public T ComputeDerivative(T r)
    {
        // Derivative with respect to r: -r/(r² + ε²)^(3/2)
        
        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);
        
        // Calculate ε²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);
        
        // Calculate r² + ε²
        T sum = _numOps.Add(rSquared, epsilonSquared);
        
        // Calculate (r² + ε²)^(3/2)
        T sumSqrt = _numOps.Sqrt(sum);
        T sumPow3_2 = _numOps.Multiply(sum, sumSqrt);
        
        // Calculate -r
        T negativeR = _numOps.Negate(r);
        
        // Return -r/(r² + ε²)^(3/2)
        return _numOps.Divide(negativeR, sumPow3_2);
    }

    public T ComputeWidthDerivative(T r)
    {
        // Derivative with respect to ε: -ε/(r² + ε²)^(3/2)
        
        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);
        
        // Calculate ε²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);
        
        // Calculate r² + ε²
        T sum = _numOps.Add(rSquared, epsilonSquared);
        
        // Calculate (r² + ε²)^(3/2)
        T sumSqrt = _numOps.Sqrt(sum);
        T sumPow3_2 = _numOps.Multiply(sum, sumSqrt);
        
        // Calculate -ε
        T negativeEpsilon = _numOps.Negate(_epsilon);
        
        // Return -ε/(r² + ε²)^(3/2)
        return _numOps.Divide(negativeEpsilon, sumPow3_2);
    }
}