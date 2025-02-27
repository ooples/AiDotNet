namespace AiDotNet.Kernels;

public class CauchyKernel<T> : IKernelFunction<T>
{
    private readonly T _sigma;
    private readonly INumericOperations<T> _numOps;

    public CauchyKernel(T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.FromDouble(1.0);
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var squaredDistance = diff.DotProduct(diff);

        return _numOps.Divide(_numOps.One, _numOps.Add(_numOps.One, _numOps.Divide(squaredDistance, _numOps.Square(_sigma))));
    }
}