namespace AiDotNet.Kernels;

public class PiecewisePolynomialKernel<T> : IKernelFunction<T>
{
    private readonly int _degree;
    private readonly T _c;
    private readonly INumericOperations<T> _numOps;

    public PiecewisePolynomialKernel(int degree = 3, T? c = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _degree = degree;
        _c = c ?? _numOps.One;
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T distance = x1.EuclideanDistance(x2);
        
        if (_numOps.GreaterThan(distance, _c))
        {
            return _numOps.Zero;
        }
        
        T j = _numOps.FromDouble(_degree);
        T term = _numOps.Subtract(_numOps.One, _numOps.Divide(distance, _c));
        
        return _numOps.Power(term, _numOps.Add(j, _numOps.One));
    }
}