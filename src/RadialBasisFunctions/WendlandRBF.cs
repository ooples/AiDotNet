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
}