namespace AiDotNet.Interpolation;

public class HermiteInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _m;
    private readonly INumericOperations<T> _numOps;

    public HermiteInterpolation(Vector<T> x, Vector<T> y, Vector<T> m)
    {
        if (x.Length != y.Length || x.Length != m.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        _x = x;
        _y = y;
        _m = m;
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
        T m0 = _m[i];
        T m1 = _m[i + 1];

        T h = _numOps.Subtract(x1, x0);
        T t = _numOps.Divide(_numOps.Subtract(x, x0), h);

        T h00 = _numOps.Multiply(_numOps.Subtract(_numOps.FromDouble(2), _numOps.Multiply(_numOps.FromDouble(3), t)), _numOps.Add(_numOps.One, _numOps.Multiply(_numOps.FromDouble(-1), t)));
        T h10 = _numOps.Multiply(h, _numOps.Multiply(t, _numOps.Add(_numOps.One, _numOps.Multiply(_numOps.FromDouble(-1), t))));
        T h01 = _numOps.Multiply(_numOps.Multiply(_numOps.FromDouble(3), t), _numOps.Subtract(_numOps.FromDouble(2), t));
        T h11 = _numOps.Multiply(h, _numOps.Multiply(t, _numOps.Subtract(t, _numOps.One)));

        return _numOps.Add(
            _numOps.Add(
                _numOps.Multiply(h00, y0),
                _numOps.Multiply(h10, m0)
            ),
            _numOps.Add(
                _numOps.Multiply(h01, y1),
                _numOps.Multiply(h11, m1)
            )
        );
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