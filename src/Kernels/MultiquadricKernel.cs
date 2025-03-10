namespace AiDotNet.Kernels;

public class MultiquadricKernel<T> : IKernelFunction<T>
{
    private readonly T _c;
    private readonly INumericOperations<T> _numOps;

    public MultiquadricKernel(T? c = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _c = c ?? _numOps.FromDouble(1.0);
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var squaredDistance = diff.DotProduct(diff);
        return _numOps.Sqrt(_numOps.Add(squaredDistance, _numOps.Multiply(_c, _c)));
    }
}