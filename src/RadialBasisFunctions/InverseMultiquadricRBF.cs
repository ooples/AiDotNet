namespace AiDotNet.RadialBasisFunctions;

public class InverseMultiquadricRBF<T> : IRadialBasisFunction<T>
{
    private readonly T _epsilon;
    private readonly INumericOperations<T> _numOps;

    public InverseMultiquadricRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    public T Compute(T r)
    {
        return _numOps.Divide(
            _numOps.One,
            _numOps.Sqrt(_numOps.Add(_numOps.Multiply(r, r), _numOps.Multiply(_epsilon, _epsilon)))
        );
    }
}