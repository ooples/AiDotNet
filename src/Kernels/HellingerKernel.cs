namespace AiDotNet.Kernels;

public class HellingerKernel<T> : IKernelFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public HellingerKernel()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            var sqrtProduct = _numOps.Sqrt(_numOps.Multiply(x1[i], x2[i]));
            sum = _numOps.Add(sum, sqrtProduct);
        }

        return sum;
    }
}