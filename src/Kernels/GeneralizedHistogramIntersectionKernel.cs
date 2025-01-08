namespace AiDotNet.Kernels;

public class GeneralizedHistogramIntersectionKernel<T> : IKernelFunction<T>
{
    private readonly T _beta;
    private readonly INumericOperations<T> _numOps;

    public GeneralizedHistogramIntersectionKernel(T? beta = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _beta = beta ?? _numOps.FromDouble(1.0);
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            sum = _numOps.Add(sum, _numOps.Power(MathHelper.Min(x1[i], x2[i]), _beta));
        }

        return sum;
    }
}