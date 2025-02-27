namespace AiDotNet.Kernels;

public class AdditiveChiSquaredKernel<T> : IKernelFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public AdditiveChiSquaredKernel()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            T numerator = _numOps.Square(_numOps.Subtract(x1[i], x2[i]));
            T denominator = _numOps.Add(x1[i], x2[i]);
            if (!_numOps.Equals(denominator, _numOps.Zero))
            {
                sum = _numOps.Add(sum, _numOps.Divide(numerator, denominator));
            }
        }

        return _numOps.Negate(_numOps.Log(_numOps.Add(_numOps.One, sum)));
    }
}