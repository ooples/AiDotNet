using System;

namespace AiDotNet.RadialBasisFunctions;

public class MaternRBF<T> : IRadialBasisFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly double _nu;
    private readonly T _lengthScale;

    public MaternRBF(double nu = 1.5, double lengthScale = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _nu = nu;
        _lengthScale = _numOps.FromDouble(lengthScale);
    }

    public T Compute(T r)
    {
        T scaledR = _numOps.Divide(r, _lengthScale);

        if (_numOps.Equals(scaledR, _numOps.Zero))
        {
            return _numOps.One;
        }

        double sqrt2nu = Math.Sqrt(2 * _nu);
        T term1 = _numOps.Power(_numOps.FromDouble(2), _numOps.FromDouble(1 - _nu));
        T term2 = _numOps.FromDouble(1 / MathHelper.Gamma(_nu));
        T term3 = _numOps.Power(_numOps.Multiply(_numOps.FromDouble(sqrt2nu), scaledR), _numOps.FromDouble(_nu));
        T term4 = _numOps.FromDouble(MathHelper.BesselK(_nu, Convert.ToDouble(_numOps.Multiply(_numOps.FromDouble(sqrt2nu), scaledR))));

        return _numOps.Multiply(_numOps.Multiply(_numOps.Multiply(term1, term2), term3), term4);
    }
}