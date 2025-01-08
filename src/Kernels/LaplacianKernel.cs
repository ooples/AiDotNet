namespace AiDotNet.Kernels;

public class LaplacianKernel<T> : IKernelFunction<T>
{
    private readonly T _sigma;
    private readonly INumericOperations<T> _numOps;

    public LaplacianKernel(T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.FromDouble(1.0);
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var distance = diff.Transform(_numOps.Abs).Sum();

        return _numOps.Exp(_numOps.Negate(_numOps.Divide(distance, _sigma)));
    }
}