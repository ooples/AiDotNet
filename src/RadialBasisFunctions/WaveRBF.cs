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
}