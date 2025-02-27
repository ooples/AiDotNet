namespace AiDotNet.Kernels;

public class BSplineKernel<T> : IKernelFunction<T>
{
    private readonly int _degree;
    private readonly T _knotSpacing;
    private readonly INumericOperations<T> _numOps;

    public BSplineKernel(int degree = 3, T? knotSpacing = default)
    {
        _degree = degree;
        _numOps = MathHelper.GetNumericOperations<T>();
        _knotSpacing = knotSpacing ?? _numOps.One;
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T result = _numOps.One;
        for (int i = 0; i < x1.Length; i++)
        {
            T diff = _numOps.Divide(_numOps.Subtract(x1[i], x2[i]), _knotSpacing);
            result = _numOps.Multiply(result, BSplineBasis(_degree, diff));
        }

        return result;
    }

    private T BSplineBasis(int degree, T x)
    {
        if (degree == 0)
        {
            return (_numOps.GreaterThanOrEquals(x, _numOps.Zero) && _numOps.LessThan(x, _numOps.One)) ? _numOps.One : _numOps.Zero;
        }

        T left = _numOps.Multiply(x, BSplineBasis(degree - 1, x));
        T right = _numOps.Multiply(_numOps.Subtract(_numOps.FromDouble(degree + 1), x), BSplineBasis(degree - 1, _numOps.Subtract(x, _numOps.One)));

        return _numOps.Divide(_numOps.Add(left, right), _numOps.FromDouble(degree));
    }
}