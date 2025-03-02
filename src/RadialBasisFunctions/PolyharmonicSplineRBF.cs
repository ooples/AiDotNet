namespace AiDotNet.RadialBasisFunctions;

public class PolyharmonicSplineRBF<T> : IRadialBasisFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _k;

    public PolyharmonicSplineRBF(int k = 2)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _k = k;
    }

    public T Compute(T r)
    {
        if (_numOps.Equals(r, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        if (_k % 2 == 0)
        {
            // For even k: r^k * log(r)
            T rPowK = _numOps.Power(r, _numOps.FromDouble(_k));
            return _numOps.Multiply(rPowK, _numOps.Log(r));
        }
        else
        {
            // For odd k: r^k
            return _numOps.Power(r, _numOps.FromDouble(_k));
        }
    }

    public T ComputeDerivative(T r)
    {
        // Handle r = 0 case
        if (_numOps.Equals(r, _numOps.Zero))
        {
            // The derivative at r = 0 depends on k
            if (_k == 1)
            {
                // For k = 1, the derivative is 1
                return _numOps.One;
            }
            else
            {
                // For k > 1, the derivative is 0
                return _numOps.Zero;
            }
        }

        if (_k % 2 == 0)
        {
            // For even k: d/dr[r^k * log(r)] = r^(k-1) * (k * log(r) + 1)
            
            // Calculate r^(k-1)
            T rPowKMinus1 = _numOps.Power(r, _numOps.FromDouble(_k - 1));
            
            // Calculate k * log(r)
            T kLogR = _numOps.Multiply(_numOps.FromDouble(_k), _numOps.Log(r));
            
            // Calculate k * log(r) + 1
            T term = _numOps.Add(kLogR, _numOps.One);
            
            // Return r^(k-1) * (k * log(r) + 1)
            return _numOps.Multiply(rPowKMinus1, term);
        }
        else
        {
            // For odd k: d/dr[r^k] = k * r^(k-1)
            
            // Calculate r^(k-1)
            T rPowKMinus1 = _numOps.Power(r, _numOps.FromDouble(_k - 1));
            
            // Return k * r^(k-1)
            return _numOps.Multiply(_numOps.FromDouble(_k), rPowKMinus1);
        }
    }

    public T ComputeWidthDerivative(T r)
    {
        // The polyharmonic spline doesn't have a width parameter,
        // so the derivative with respect to width is 0
        return _numOps.Zero;
    }
}