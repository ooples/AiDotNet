namespace AiDotNet.Interpolation;

public class CatmullRomSplineInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly T _tension;
    private readonly INumericOperations<T> _numOps;

    public CatmullRomSplineInterpolation(Vector<T> x, Vector<T> y, double tension = 0.5)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 4)
            throw new ArgumentException("At least 4 points are required for Catmull-Rom spline interpolation.");

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
        _tension = _numOps.FromDouble(tension);
    }

    public T Interpolate(T x)
    {
        int i = FindInterval(x);

        T x0 = _x[i - 1];
        T x1 = _x[i];
        T x2 = _x[i + 1];
        T x3 = _x[i + 2];

        T y0 = _y[i - 1];
        T y1 = _y[i];
        T y2 = _y[i + 1];
        T y3 = _y[i + 2];

        T t = _numOps.Divide(_numOps.Subtract(x, x1), _numOps.Subtract(x2, x1));

        return CatmullRomSpline(y0, y1, y2, y3, t);
    }

    private int FindInterval(T x)
    {
        for (int i = 1; i < _x.Length - 2; i++)
        {
            if (_numOps.LessThanOrEquals(x, _x[i + 1]))
            {
                return i;
            }
        }
        return _x.Length - 3;
    }

    private T CatmullRomSpline(T y0, T y1, T y2, T y3, T t)
    {
        T t2 = _numOps.Multiply(t, t);
        T t3 = _numOps.Multiply(t2, t);

        T c0 = _numOps.Multiply(_numOps.FromDouble(-0.5), _numOps.Add(_numOps.Multiply(_tension, y0), _numOps.Multiply(_tension, y2)));
        T c1 = _numOps.Add(_numOps.Multiply(_numOps.FromDouble(1.5), y1), _numOps.Multiply(_numOps.FromDouble(-1.5), y2));
        T c2 = _numOps.Add(
            _numOps.Add(
                _numOps.Multiply(_numOps.FromDouble(2), y2),
                _numOps.Multiply(_numOps.FromDouble(-2), y1)
            ),
            _numOps.Add(
                _numOps.Multiply(_numOps.Multiply(_numOps.FromDouble(0.5), _tension), y0),
                _numOps.Multiply(_numOps.Multiply(_numOps.FromDouble(0.5), _tension), y3)
            )
        );
        T c3 = _numOps.Add(
            _numOps.Add(
                _numOps.Multiply(_numOps.FromDouble(-0.5), y2),
                _numOps.Multiply(_numOps.FromDouble(0.5), y1)
            ),
            _numOps.Add(
                _numOps.Multiply(_numOps.Multiply(_numOps.FromDouble(-0.25), _tension), y0),
                _numOps.Multiply(_numOps.Multiply(_numOps.FromDouble(0.25), _tension), y3)
            )
        );

        return _numOps.Add(
            _numOps.Add(
                _numOps.Multiply(c3, t3),
                _numOps.Multiply(c2, t2)
            ),
            _numOps.Add(
                _numOps.Multiply(c1, t),
                y1
            )
        );
    }
}