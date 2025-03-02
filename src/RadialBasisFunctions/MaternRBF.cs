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

    public T ComputeDerivative(T r)
    {
        // For r = 0, the derivative is 0 due to symmetry
        if (_numOps.Equals(r, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        T scaledR = _numOps.Divide(r, _lengthScale);
        double sqrt2nu = Math.Sqrt(2 * _nu);
        T sqrtTerm = _numOps.FromDouble(sqrt2nu);
        T x = _numOps.Multiply(sqrtTerm, scaledR); // √(2ν)r/l

        // Common terms from the original function
        T term1 = _numOps.Power(_numOps.FromDouble(2), _numOps.FromDouble(1 - _nu));
        T term2 = _numOps.FromDouble(1 / MathHelper.Gamma(_nu));
        
        // For special case ν = 0.5, the derivative has a simpler form
        if (Math.Abs(_nu - 0.5) < 1e-10)
        {
            // For ν = 0.5, K_0.5(x) = √(π/2x) * e^(-x)
            // The derivative simplifies considerably
            T expTerm = _numOps.Exp(_numOps.Negate(x));
            return _numOps.Multiply(
                _numOps.Negate(_numOps.Divide(sqrtTerm, _lengthScale)),
                expTerm
            );
        }
        
        // For special case ν = 1.5, the derivative also has a simpler form
        if (Math.Abs(_nu - 1.5) < 1e-10)
        {
            // For ν = 1.5, we can use a simplified formula
            T expTerm = _numOps.Exp(_numOps.Negate(x));
            T factor = _numOps.Multiply(
                _numOps.FromDouble(sqrt2nu / Convert.ToDouble(_lengthScale)),
                _numOps.Divide(
                    _numOps.Add(_numOps.One, x),
                    x
                )
            );
            return _numOps.Multiply(_numOps.Negate(factor), expTerm);
        }
        
        // For general case, we need to use the recurrence relation for Bessel functions
        // d/dr[K_ν(x)] = -K_(ν-1)(x) - (ν/x)K_ν(x) where x = √(2ν)r/l
        
        double xDouble = Convert.ToDouble(x);
        double besselKnu = MathHelper.BesselK(_nu, xDouble);
        double besselKnuMinus1 = MathHelper.BesselK(_nu - 1, xDouble);
        
        // Calculate d/dx[K_ν(x)]
        T dBesselK = _numOps.Add(
            _numOps.Negate(_numOps.FromDouble(besselKnuMinus1)),
            _numOps.Multiply(
                _numOps.Negate(_numOps.FromDouble(_nu / xDouble)),
                _numOps.FromDouble(besselKnu)
            )
        );
        
        // Calculate d/dr[x] = √(2ν)/l
        T dxdr = _numOps.Divide(sqrtTerm, _lengthScale);
        
        // Calculate d/dr[x^ν] = ν*x^(ν-1) * d/dr[x]
        T dxPowerNu = _numOps.Multiply(
            _numOps.Multiply(
                _numOps.FromDouble(_nu),
                _numOps.Power(x, _numOps.FromDouble(_nu - 1))
            ),
            dxdr
        );
        
        // Apply product rule: d/dr[x^ν * K_ν(x)] = x^ν * d/dr[K_ν(x)] + K_ν(x) * d/dr[x^ν]
        T term3 = _numOps.Power(x, _numOps.FromDouble(_nu));
        T term4 = _numOps.FromDouble(besselKnu);
        
        T productRule = _numOps.Add(
            _numOps.Multiply(term3, _numOps.Multiply(dBesselK, dxdr)),
            _numOps.Multiply(term4, dxPowerNu)
        );
        
        // Combine with the constant terms
        return _numOps.Multiply(_numOps.Multiply(term1, term2), productRule);
    }

    public T ComputeWidthDerivative(T r)
    {
        // For r = 0, the width derivative is 0
        if (_numOps.Equals(r, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        T scaledR = _numOps.Divide(r, _lengthScale);
        double sqrt2nu = Math.Sqrt(2 * _nu);
        T sqrtTerm = _numOps.FromDouble(sqrt2nu);
        T x = _numOps.Multiply(sqrtTerm, scaledR); // √(2ν)r/l

        // Common terms from the original function
        T term1 = _numOps.Power(_numOps.FromDouble(2), _numOps.FromDouble(1 - _nu));
        T term2 = _numOps.FromDouble(1 / MathHelper.Gamma(_nu));
        T term3 = _numOps.Power(x, _numOps.FromDouble(_nu));
        
        double xDouble = Convert.ToDouble(x);
        double besselKnu = MathHelper.BesselK(_nu, xDouble);
        T term4 = _numOps.FromDouble(besselKnu);
        
        // The width derivative involves d/dl[x] = -√(2ν)r/l²
        T dxdl = _numOps.Negate(_numOps.Divide(x, _lengthScale));
        
        // For special case ν = 0.5, the width derivative has a simpler form
        if (Math.Abs(_nu - 0.5) < 1e-10)
        {
            // For ν = 0.5, we can use a simplified formula
            T expTerm = _numOps.Exp(_numOps.Negate(x));
            return _numOps.Multiply(x, _numOps.Multiply(dxdl, expTerm));
        }
        
        // For special case ν = 1.5, the width derivative also has a simpler form
        if (Math.Abs(_nu - 1.5) < 1e-10)
        {
            // For ν = 1.5, we can use a simplified formula
            T expTerm = _numOps.Exp(_numOps.Negate(x));
            T factor = _numOps.Multiply(
                x,
                _numOps.Divide(
                    _numOps.Add(_numOps.One, x),
                    x
                )
            );
            return _numOps.Multiply(factor, _numOps.Multiply(dxdl, expTerm));
        }
        
        // For general case, we need to use the recurrence relation for Bessel functions
        double besselKnuMinus1 = MathHelper.BesselK(_nu - 1, xDouble);
        
        // Calculate d/dx[K_ν(x)]
        T dBesselK = _numOps.Add(
            _numOps.Negate(_numOps.FromDouble(besselKnuMinus1)),
            _numOps.Multiply(
                _numOps.Negate(_numOps.FromDouble(_nu / xDouble)),
                term4
            )
        );
        
        // Calculate d/dl[x^ν] = ν*x^(ν-1) * d/dl[x]
        T dxPowerNu = _numOps.Multiply(
            _numOps.Multiply(
                _numOps.FromDouble(_nu),
                _numOps.Power(x, _numOps.FromDouble(_nu - 1))
            ),
            dxdl
        );
        
        // Apply product rule: d/dl[x^ν * K_ν(x)] = x^ν * d/dl[K_ν(x)] + K_ν(x) * d/dl[x^ν]
        T productRule = _numOps.Add(
            _numOps.Multiply(term3, _numOps.Multiply(dBesselK, dxdl)),
            _numOps.Multiply(term4, dxPowerNu)
        );
        
        // Combine with the constant terms
        return _numOps.Multiply(_numOps.Multiply(term1, term2), productRule);
    }
}