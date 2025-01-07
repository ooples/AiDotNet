namespace AiDotNet.Kernels;

public class SigmoidKernel<T> : IKernelFunction<T>
{
    private readonly T _alpha;
    private readonly T _c;
    private readonly INumericOperations<T> _numOps;

    public SigmoidKernel(T alpha, T c)
    {
        _alpha = alpha;
        _c = c;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var dotProduct = x1.DotProduct(x2);
        return MathHelper.Tanh(_numOps.Add(_numOps.Multiply(_alpha, dotProduct), _c));
    }
}