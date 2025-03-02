namespace AiDotNet.RadialBasisFunctions;

public class MultiquadricRBF<T> : IRadialBasisFunction<T>
{
    private readonly T _epsilon;
    private readonly INumericOperations<T> _numOps;

    public MultiquadricRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    public T Compute(T r)
    {
        return _numOps.Sqrt(_numOps.Add(_numOps.Multiply(r, r), _numOps.Multiply(_epsilon, _epsilon)));
    }

    public T ComputeDerivative(T r)
    {
        // Derivative with respect to r: r/√(r² + ε²)
        
        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);
        
        // Calculate ε²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);
        
        // Calculate r² + ε²
        T sum = _numOps.Add(rSquared, epsilonSquared);
        
        // Calculate √(r² + ε²)
        T sqrtSum = _numOps.Sqrt(sum);
        
        // Return r/√(r² + ε²)
        return _numOps.Divide(r, sqrtSum);
    }

    public T ComputeWidthDerivative(T r)
    {
        // Derivative with respect to ε: ε/√(r² + ε²)
        
        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);
        
        // Calculate ε²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);
        
        // Calculate r² + ε²
        T sum = _numOps.Add(rSquared, epsilonSquared);
        
        // Calculate √(r² + ε²)
        T sqrtSum = _numOps.Sqrt(sum);
        
        // Return ε/√(r² + ε²)
        return _numOps.Divide(_epsilon, sqrtSum);
    }
}