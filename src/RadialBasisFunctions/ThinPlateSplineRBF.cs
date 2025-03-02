namespace AiDotNet.RadialBasisFunctions;

public class ThinPlateSplineRBF<T> : IRadialBasisFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public ThinPlateSplineRBF()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Compute(T r)
    {
        if (_numOps.Equals(r, _numOps.Zero))
        {
            return _numOps.Zero;
        }
        
        T r2 = _numOps.Multiply(r, r);
        return _numOps.Multiply(r2, _numOps.Log(r));
    }

    public T ComputeDerivative(T r)
    {
        // For r = 0, the derivative is 0
        if (_numOps.Equals(r, _numOps.Zero))
        {
            return _numOps.Zero;
        }
        
        // Calculate log(r)
        T logR = _numOps.Log(r);
        
        // Calculate 2 * log(r)
        T twoLogR = _numOps.Multiply(_numOps.FromDouble(2.0), logR);
        
        // Calculate 2 * log(r) + 1
        T term = _numOps.Add(twoLogR, _numOps.One);
        
        // Return r * (2 * log(r) + 1)
        return _numOps.Multiply(r, term);
    }

    public T ComputeWidthDerivative(T r)
    {
        // The thin plate spline doesn't have a width parameter,
        // so the derivative with respect to width is 0
        return _numOps.Zero;
    }
}