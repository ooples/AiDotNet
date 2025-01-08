namespace AiDotNet.Kernels;

public class HistogramIntersectionKernel<T> : IKernelFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public HistogramIntersectionKernel()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            sum = _numOps.Add(sum, MathHelper.Min(x1[i], x2[i]));
        }

        return sum;
    }
}