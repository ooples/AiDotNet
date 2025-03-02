namespace AiDotNet.RadialBasisFunctions;

public class WendlandRBF<T> : IRadialBasisFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _k;
    private readonly T _supportRadius;

    public WendlandRBF(int k = 2, double supportRadius = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _k = k;
        _supportRadius = _numOps.FromDouble(supportRadius);
    }

    public T Compute(T r)
    {
        T normalizedR = _numOps.Divide(r, _supportRadius);
        
        if (_numOps.GreaterThanOrEquals(normalizedR, _numOps.One))
        {
            return _numOps.Zero;
        }

        T oneMinusR = _numOps.Subtract(_numOps.One, normalizedR);

        switch (_k)
        {
            case 0:
                return _numOps.Power(oneMinusR, _numOps.FromDouble(2));
            case 1:
                T term1 = _numOps.Power(oneMinusR, _numOps.FromDouble(4));
                T term2 = _numOps.Multiply(_numOps.FromDouble(4), normalizedR);
                return _numOps.Multiply(term1, _numOps.Add(_numOps.One, term2));
            case 2:
                T term1_k2 = _numOps.Power(oneMinusR, _numOps.FromDouble(6));
                T term2_k2 = _numOps.Multiply(_numOps.FromDouble(35), _numOps.Power(normalizedR, _numOps.FromDouble(2)));
                T term3_k2 = _numOps.Multiply(_numOps.FromDouble(18), normalizedR);
                T term4_k2 = _numOps.FromDouble(3);
                return _numOps.Multiply(term1_k2, _numOps.Add(_numOps.Add(term2_k2, term3_k2), term4_k2));
            default:
                throw new ArgumentException("Unsupported k value. Supported values are 0, 1, and 2.");
        }
    }

    public T ComputeDerivative(T r)
    {
        T normalizedR = _numOps.Divide(r, _supportRadius);
    
        if (_numOps.GreaterThanOrEquals(normalizedR, _numOps.One) || _numOps.Equals(normalizedR, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        T oneMinusR = _numOps.Subtract(_numOps.One, normalizedR);

        switch (_k)
        {
            case 0:
                // d/dr[(1-r)^2] = -2(1-r)
                T factor = _numOps.FromDouble(-2.0);
                return _numOps.Multiply(factor, oneMinusR);
            
            case 1:
                // d/dr[(1-r)^4 * (1+4r)] = (1-r)^3 * (-4-20r)
                T term1 = _numOps.Power(oneMinusR, _numOps.FromDouble(3));
                T term2 = _numOps.FromDouble(-4);
                T term3 = _numOps.Multiply(_numOps.FromDouble(-20), normalizedR);
                return _numOps.Multiply(term1, _numOps.Add(term2, term3));
            
            case 2:
                // d/dr[(1-r)^6 * (3+18r+35r^2)] = (1-r)^5 * (-18-180r-210r^2)
                T term1_k2 = _numOps.Power(oneMinusR, _numOps.FromDouble(5));
                T term2_k2 = _numOps.FromDouble(-18);
                T term3_k2 = _numOps.Multiply(_numOps.FromDouble(-180), normalizedR);
                T term4_k2 = _numOps.Multiply(_numOps.FromDouble(-210), _numOps.Power(normalizedR, _numOps.FromDouble(2)));
                return _numOps.Multiply(term1_k2, _numOps.Add(_numOps.Add(term2_k2, term3_k2), term4_k2));
            
            default:
                throw new ArgumentException("Unsupported k value. Supported values are 0, 1, and 2.");
        }
    }

    public T ComputeWidthDerivative(T r)
    {
        T normalizedR = _numOps.Divide(r, _supportRadius);
    
        if (_numOps.GreaterThanOrEquals(normalizedR, _numOps.One))
        {
            return _numOps.Zero;
        }

        // For width derivative, we need to compute d/dσ[φ(r/σ)]
        // This equals -r/σ^2 * φ'(r/σ) where φ' is the derivative of φ
    
        // First, compute r/σ^2
        T rOverSigmaSquared = _numOps.Divide(r, _numOps.Power(_supportRadius, _numOps.FromDouble(2)));
    
        // Then compute the derivative at r/σ
        T derivativeValue = ComputeDerivative(r);
    
        // Multiply by -1
        T negativeOne = _numOps.FromDouble(-1);
    
        // Return -r/σ^2 * φ'(r/σ)
        return _numOps.Multiply(negativeOne, _numOps.Multiply(rOverSigmaSquared, derivativeValue));
    }
}