namespace AiDotNet.Interpolation;

public class LinearInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly INumericOperations<T> _numOps;

    public LinearInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Interpolate(T x)
    {
        int i = FindInterval(x);

        if (i == _x.Length - 1)
        {
            return _y[i];
        }

        T x0 = _x[i];
        T x1 = _x[i + 1];
        T y0 = _y[i];
        T y1 = _y[i + 1];

        T t = _numOps.Divide(_numOps.Subtract(x, x0), _numOps.Subtract(x1, x0));
        return _numOps.Add(_numOps.Multiply(_numOps.Subtract(_numOps.One, t), y0), _numOps.Multiply(t, y1));
    }

    private int FindInterval(T x)
    {
        if (_numOps.LessThanOrEquals(x, _x[0]))
            return 0;
        if (_numOps.GreaterThanOrEquals(x, _x[_x.Length - 1]))
            return _x.Length - 1;

        int low = 0;
        int high = _x.Length - 1;

        while (low < high - 1)
        {
            int mid = (low + high) / 2;
            if (_numOps.LessThanOrEquals(_x[mid], x))
                low = mid;
            else
                high = mid;
        }

        return low;
    }
}