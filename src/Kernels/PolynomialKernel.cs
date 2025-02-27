namespace AiDotNet.Kernels;

public class PolynomialKernel<T> : IKernelFunction<T>
{
    private readonly T _degree;
    private readonly T _coef0;
    private readonly INumericOperations<T> _numOps;

    public PolynomialKernel(T? degree = default, T? coef0 = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _degree = degree ?? _numOps.FromDouble(3.0);
        _coef0 = coef0 ?? _numOps.FromDouble(1.0);
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var dotProduct = x1.DotProduct(x2);
        return _numOps.Power(_numOps.Add(dotProduct, _coef0), _degree);
    }
}