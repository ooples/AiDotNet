namespace AiDotNet.Kernels;

public class SphericalKernel<T> : IKernelFunction<T>
{
    private readonly T _sigma;
    private readonly INumericOperations<T> _numOps;

    public SphericalKernel(T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.FromDouble(1.0);
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var distance = _numOps.Sqrt(diff.DotProduct(diff));
        var normalizedDistance = _numOps.Divide(distance, _sigma);

        if (_numOps.GreaterThan(normalizedDistance, _numOps.One))
        {
            return _numOps.Zero;
        }

        var oneMinusR = _numOps.Subtract(_numOps.One, normalizedDistance);
        return _numOps.Multiply(_numOps.FromDouble(1.5), oneMinusR);
    }
}