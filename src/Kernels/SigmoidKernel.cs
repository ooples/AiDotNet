namespace AiDotNet.Kernels;

public class SigmoidKernel<T> : IKernelFunction<T>
{
    private readonly T _alpha;
    private readonly T _c;
    private readonly INumericOperations<T> _numOps;

    public SigmoidKernel(T? alpha = default, T? c = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _alpha = alpha ?? _numOps.FromDouble(1.0);
        _c = c ?? _numOps.FromDouble(0.0);
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var dotProduct = x1.DotProduct(x2);
        return MathHelper.Tanh(_numOps.Add(_numOps.Multiply(_alpha, dotProduct), _c));
    }
}