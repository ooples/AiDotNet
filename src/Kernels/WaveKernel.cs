namespace AiDotNet.Kernels;

public class WaveKernel<T> : IKernelFunction<T>
{
    private readonly T _sigma;
    private readonly INumericOperations<T> _numOps;

    public WaveKernel(T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.FromDouble(1.0);
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var distance = _numOps.Sqrt(diff.DotProduct(diff));
        var normalizedDistance = _numOps.Divide(distance, _sigma);

        if (_numOps.Equals(normalizedDistance, _numOps.Zero))
        {
            return _numOps.One;
        }

        return _numOps.Divide(MathHelper.Sin(normalizedDistance), normalizedDistance);
    }
}