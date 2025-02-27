namespace AiDotNet.Kernels;

public class PowerKernel<T> : IKernelFunction<T>
{
    private readonly T _degree;
    private readonly INumericOperations<T> _numOps;

    public PowerKernel(T? degree = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _degree = degree ?? _numOps.FromDouble(2.0);
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var distance = _numOps.Sqrt(diff.DotProduct(diff));

        return _numOps.Negate(_numOps.Power(distance, _degree));
    }
}