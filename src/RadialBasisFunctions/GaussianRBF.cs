namespace AiDotNet.RadialBasisFunctions;

public class GaussianRBF<T> : IRadialBasisFunction<T>
{
    private readonly T _epsilon;
    private readonly INumericOperations<T> _numOps;

    public GaussianRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    public T Compute(T r)
    {
        return _numOps.Exp(_numOps.Negate(_numOps.Multiply(_epsilon, _numOps.Multiply(r, r))));
    }

    public T ComputeDerivative(T r)
    {
        // Derivative with respect to r: -2εr * exp(-ε*r²)
        
        // Calculate -2εr
        T minusTwoEpsilonR = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(-2.0), _epsilon), 
            r
        );
        
        // Multiply by exp(-ε*r²)
        return _numOps.Multiply(minusTwoEpsilonR, Compute(r));
    }

    public T ComputeWidthDerivative(T r)
    {
        // Derivative with respect to ε: -r² * exp(-ε*r²)
        
        // Calculate -r²
        T rSquared = _numOps.Multiply(r, r);
        T negativeRSquared = _numOps.Negate(rSquared);
        
        // Multiply by exp(-ε*r²)
        return _numOps.Multiply(negativeRSquared, Compute(r));
    }
}