namespace AiDotNet.Kernels;

public class TanimotoKernel<T> : IKernelFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public TanimotoKernel()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T dotProduct = x1.DotProduct(x2);
        T x1Norm = x1.DotProduct(x1);
        T x2Norm = x2.DotProduct(x2);

        return _numOps.Divide(dotProduct, _numOps.Subtract(_numOps.Add(x1Norm, x2Norm), dotProduct));
    }
}