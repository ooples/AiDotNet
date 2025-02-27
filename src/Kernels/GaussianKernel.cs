namespace AiDotNet.Kernels;

public class GaussianKernel<T> : IKernelFunction<T>
{
    private readonly double _sigma;
    private readonly INumericOperations<T> _numOps;

    public GaussianKernel(double sigma = 1.0)
    {
        _sigma = sigma;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        T squaredDistance = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            T diff = _numOps.Subtract(x1[i], x2[i]);
            squaredDistance = _numOps.Add(squaredDistance, _numOps.Multiply(diff, diff));
        }

        T exponent = _numOps.Divide(squaredDistance, _numOps.FromDouble(2 * _sigma * _sigma));
        return _numOps.Exp(_numOps.Negate(exponent));
    }
}