namespace AiDotNet.Kernels;

public class LocallyPeriodicKernel<T> : IKernelFunction<T>
{
    private readonly T _lengthScale;
    private readonly T _period;
    private readonly T _amplitude;
    private readonly INumericOperations<T> _numOps;

    public LocallyPeriodicKernel(T? lengthScale = default, T? period = default, T? amplitude = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _lengthScale = lengthScale ?? _numOps.One;
        _period = period ?? _numOps.FromDouble(2 * Math.PI);
        _amplitude = amplitude ?? _numOps.One;
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T distance = x1.EuclideanDistance(x2);
        T periodicTerm = MathHelper.Sin(_numOps.Divide(_numOps.Multiply(MathHelper.Pi<T>(), distance), _period));
        T squaredPeriodicTerm = _numOps.Square(periodicTerm);
        T expTerm = _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2), squaredPeriodicTerm), _numOps.Square(_lengthScale))));

        return _numOps.Multiply(_numOps.Square(_amplitude), expTerm);
    }
}