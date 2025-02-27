namespace AiDotNet.Kernels;

public class ExponentialKernel<T> : IKernelFunction<T>
{
    private readonly T _sigma;
    private readonly INumericOperations<T> _numOps;

    public ExponentialKernel(T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.FromDouble(1.0);
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var distance = _numOps.Sqrt(diff.DotProduct(diff));

        return _numOps.Exp(_numOps.Divide(_numOps.Negate(distance), _sigma));
    }
}