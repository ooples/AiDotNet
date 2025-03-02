namespace AiDotNet.RadialBasisFunctions;

public class WaveRBF<T> : IRadialBasisFunction<T>
{
    private readonly T _epsilon;
    private readonly INumericOperations<T> _numOps;

    public WaveRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    public T Compute(T r)
    {
        T epsilonR = _numOps.Multiply(_epsilon, r);
        
        // Handle the case when epsilonR is very close to zero
        if (MathHelper.AlmostEqual(epsilonR, _numOps.Zero))
        {
            return _numOps.One;
        }

        T sinEpsilonR = MathHelper.Sin(epsilonR);
        return _numOps.Divide(sinEpsilonR, epsilonR);
    }

    public T ComputeDerivative(T r)
    {
        T epsilonR = _numOps.Multiply(_epsilon, r);
        
        // Handle the case when epsilonR is very close to zero
        if (MathHelper.AlmostEqual(epsilonR, _numOps.Zero))
        {
            // For εr → 0, the derivative approaches 0
            return _numOps.Zero;
        }
        
        // Calculate cos(εr)
        T cosEpsilonR = MathHelper.Cos(epsilonR);
        
        // Calculate sin(εr)
        T sinEpsilonR = MathHelper.Sin(epsilonR);
        
        // Calculate ε·r·cos(εr)
        T epsilonRCosEpsilonR = _numOps.Multiply(epsilonR, cosEpsilonR);
        
        // Calculate ε·r·cos(εr) + sin(εr)
        T numerator = _numOps.Add(epsilonRCosEpsilonR, sinEpsilonR);
        
        // Calculate (εr)²
        T epsilonRSquared = _numOps.Multiply(epsilonR, epsilonR);
        
        // Return (ε·r·cos(εr) + sin(εr))/(εr)²
        return _numOps.Divide(numerator, epsilonRSquared);
    }

    public T ComputeWidthDerivative(T r)
    {
        T epsilonR = _numOps.Multiply(_epsilon, r);
        T rSquared = _numOps.Multiply(r, r);
        
        // Handle the case when epsilonR is very close to zero
        if (MathHelper.AlmostEqual(epsilonR, _numOps.Zero))
        {
            // For εr → 0, the width derivative approaches -r²/3
            T negativeRSquaredDivThree = _numOps.Divide(
                _numOps.Negate(rSquared),
                _numOps.FromDouble(3.0)
            );
            return negativeRSquaredDivThree;
        }
        
        // Calculate cos(εr)
        T cosEpsilonR = MathHelper.Cos(epsilonR);
        
        // Calculate sin(εr)
        T sinEpsilonR = MathHelper.Sin(epsilonR);
        
        // Calculate r²·cos(εr)
        T rSquaredCosEpsilonR = _numOps.Multiply(rSquared, cosEpsilonR);
        
        // Calculate sin(εr)/ε
        T sinEpsilonRDivEpsilon = _numOps.Divide(sinEpsilonR, _epsilon);
        
        // Calculate r²·cos(εr) + sin(εr)/ε
        T numerator = _numOps.Add(rSquaredCosEpsilonR, sinEpsilonRDivEpsilon);
        
        // Calculate (εr)²
        T epsilonRSquared = _numOps.Multiply(epsilonR, epsilonR);
        
        // Return (r²·cos(εr) + sin(εr)/ε)/(εr)²
        return _numOps.Divide(numerator, epsilonRSquared);
    }
}