namespace AiDotNet.Kernels;

public class SplineKernel<T> : IKernelFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public SplineKernel()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

   public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T result = _numOps.One;
        for (int i = 0; i < x1.Length; i++)
        {
            T min = MathHelper.Min(x1[i], x2[i]);
            T product = _numOps.Multiply(x1[i], x2[i]);
            T minProduct = _numOps.Multiply(min, product);
            T minCubed = _numOps.Power(min, _numOps.FromDouble(3));
            T halfMinCubed = _numOps.Multiply(_numOps.FromDouble(0.5), minCubed);
        
            T innerSum = _numOps.Add(_numOps.One, minProduct);
            innerSum = _numOps.Add(innerSum, halfMinCubed);
        
            result = _numOps.Multiply(result, innerSum);
        }

        return result;
    }
}