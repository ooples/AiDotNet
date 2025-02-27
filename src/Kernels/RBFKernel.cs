namespace AiDotNet.Kernels;

public class RBFKernel<T> : IKernelFunction<T>
{
    private readonly T _gamma;
    private readonly INumericOperations<T> _numOps;

    public RBFKernel(T? gamma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _gamma = gamma ?? _numOps.FromDouble(1.0);
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var squaredDistance = diff.DotProduct(diff);

        return _numOps.Exp(_numOps.Multiply(_numOps.Negate(_gamma), squaredDistance));
    }
}