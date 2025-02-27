namespace AiDotNet.Kernels;

public class ChiSquareKernel<T> : IKernelFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public ChiSquareKernel()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            var diff = _numOps.Subtract(x1[i], x2[i]);
            var numerator = _numOps.Square(diff);
            var denominator = _numOps.Add(x1[i], x2[i]);
        
            if (!_numOps.Equals(denominator, _numOps.Zero))
            {
                sum = _numOps.Add(sum, _numOps.Divide(numerator, denominator));
            }
        }

        return _numOps.Subtract(_numOps.One, sum);
    }
}