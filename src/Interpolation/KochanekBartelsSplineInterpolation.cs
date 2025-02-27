namespace AiDotNet.Interpolation;

public class KochanekBartelsSplineInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly T _tension;
    private readonly T _bias;
    private readonly T _continuity;
    private readonly INumericOperations<T> _numOps;

    public KochanekBartelsSplineInterpolation(Vector<T> x, Vector<T> y, double tension = 0, double bias = 0, double continuity = 0)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 4)
            throw new ArgumentException("At least 4 points are required for Kochanek–Bartels spline interpolation.");

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
        _tension = _numOps.FromDouble(tension);
        _bias = _numOps.FromDouble(bias);
        _continuity = _numOps.FromDouble(continuity);
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

        return KochanekBartelsSpline(y0, y1, y2, y3, t);
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

    private T KochanekBartelsSpline(T y0, T y1, T y2, T y3, T t)
    {
        T t2 = _numOps.Multiply(t, t);
        T t3 = _numOps.Multiply(t2, t);

        T m1 = CalculateTangent(y0, y1, y2);
        T m2 = CalculateTangent(y1, y2, y3);

        T c0 = y1;
        T c1 = m1;
        T c2 = _numOps.Subtract(
            _numOps.Subtract(
                _numOps.Multiply(_numOps.FromDouble(3), _numOps.Subtract(y2, y1)),
                _numOps.Multiply(_numOps.FromDouble(2), m1)
            ),
            m2
        );
        T c3 = _numOps.Add(
            _numOps.Add(
                _numOps.Multiply(_numOps.FromDouble(2), _numOps.Subtract(y1, y2)),
                m1
            ),
            m2
        );

        return _numOps.Add(
            _numOps.Add(
                _numOps.Multiply(c3, t3),
                _numOps.Multiply(c2, t2)
            ),
            _numOps.Add(
                _numOps.Multiply(c1, t),
                c0
            )
        );
    }

    private T CalculateTangent(T y0, T y1, T y2)
    {
        T oneMinusTension = _numOps.Subtract(_numOps.One, _tension);
        T oneMinusContinuity = _numOps.Subtract(_numOps.One, _continuity);
        T onePlusContinuity = _numOps.Add(_numOps.One, _continuity);
        T oneMinusBias = _numOps.Subtract(_numOps.One, _bias);
        T onePlusBias = _numOps.Add(_numOps.One, _bias);

        T a = _numOps.Multiply(
            _numOps.Multiply(
                _numOps.Multiply(_numOps.FromDouble(0.5), oneMinusTension),
                oneMinusContinuity
            ),
            onePlusBias
        );
        T b = _numOps.Multiply(
            _numOps.Multiply(
                _numOps.Multiply(_numOps.FromDouble(0.5), oneMinusTension),
                onePlusContinuity
            ),
            oneMinusBias
        );

        return _numOps.Add(
            _numOps.Multiply(a, _numOps.Subtract(y1, y0)),
            _numOps.Multiply(b, _numOps.Subtract(y2, y1))
        );
    }
}