namespace AiDotNet.Kernels;

public class LinearKernel<T> : IKernelFunction<T>
{
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        return x1.DotProduct(x2);
    }
}