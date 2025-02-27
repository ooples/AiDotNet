namespace AiDotNet.Kernels;

public class GeneralizedTStudentKernel<T> : IKernelFunction<T>
{
    private readonly T _degree;
    private readonly INumericOperations<T> _numOps;

    public GeneralizedTStudentKernel(T? degree = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _degree = degree ?? _numOps.FromDouble(1.0);
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var squaredDistance = diff.DotProduct(diff);

        return _numOps.Divide(_numOps.One, _numOps.Add(_numOps.One, _numOps.Power(squaredDistance, _degree)));
    }
}