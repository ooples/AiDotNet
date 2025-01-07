namespace AiDotNet.Kernels;

public class RationalQuadraticKernel<T> : IKernelFunction<T>
{
    private readonly T _c;
    private readonly INumericOperations<T> _numOps;

    public RationalQuadraticKernel(T c)
    {
        _c = c;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var squaredDistance = diff.DotProduct(diff);
        return _numOps.Subtract(_numOps.One, _numOps.Divide(squaredDistance, _numOps.Add(squaredDistance, _c)));
    }
}