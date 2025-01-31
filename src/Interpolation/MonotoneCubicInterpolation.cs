namespace AiDotNet.Interpolation;

public class MonotoneCubicInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _m;
    private readonly INumericOperations<T> _numOps;

    public MonotoneCubicInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        if (x.Length < 2)
        {
            throw new ArgumentException("Monotone cubic interpolation requires at least 2 points.");
        }

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
        _m = new Vector<T>(x.Length);

        CalculateSlopes();
    }

    public T Interpolate(T x)
    {
        int i = FindInterval(x);

        T h = _numOps.Subtract(_x[i + 1], _x[i]);
        T t = _numOps.Divide(_numOps.Subtract(x, _x[i]), h);

        T t2 = _numOps.Multiply(t, t);
        T t3 = _numOps.Multiply(t2, t);

        T h00 = _numOps.Add(_numOps.Multiply(_numOps.FromDouble(2), t3), _numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(-3), t2), _numOps.FromDouble(1)));
        T h10 = _numOps.Add(t3, _numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(-2), t2), t));
        T h01 = _numOps.Add(_numOps.Multiply(_numOps.FromDouble(-2), t3), _numOps.Multiply(_numOps.FromDouble(3), t2));
        T h11 = _numOps.Subtract(t3, t2);

        return _numOps.Add(
            _numOps.Add(
                _numOps.Multiply(h00, _y[i]),
                _numOps.Multiply(h10, _numOps.Multiply(h, _m[i]))
            ),
            _numOps.Add(
                _numOps.Multiply(h01, _y[i + 1]),
                _numOps.Multiply(h11, _numOps.Multiply(h, _m[i + 1]))
            )
        );
    }

    private void CalculateSlopes()
    {
        int n = _x.Length;

        // Calculate secant slopes
        Vector<T> delta = new Vector<T>(n - 1);
        for (int i = 0; i < n - 1; i++)
        {
            delta[i] = _numOps.Divide(
                _numOps.Subtract(_y[i + 1], _y[i]),
                _numOps.Subtract(_x[i + 1], _x[i])
            );
        }

        // Initialize slopes
        _m[0] = delta[0];
        _m[n - 1] = delta[n - 2];

        // Calculate interior slopes
        for (int i = 1; i < n - 1; i++)
        {
            T m = _numOps.Divide(_numOps.Add(delta[i - 1], delta[i]), _numOps.FromDouble(2));

            // Ensure monotonicity
            if (!_numOps.Equals(delta[i - 1], _numOps.Zero) && !_numOps.Equals(delta[i], _numOps.Zero))
            {
                T alpha = _numOps.Divide(delta[i - 1], delta[i]);
                T beta = _numOps.Divide(_numOps.FromDouble(3), _numOps.Add(_numOps.FromDouble(2), alpha));

                if (_numOps.GreaterThan(_numOps.Multiply(beta, m), delta[i - 1]))
                {
                    m = _numOps.Divide(delta[i - 1], beta);
                }
                else if (_numOps.GreaterThan(_numOps.Multiply(beta, m), delta[i]))
                {
                    m = _numOps.Divide(delta[i], beta);
                }
            }

            _m[i] = m;
        }
    }

    private int FindInterval(T x)
    {
        if (_numOps.LessThanOrEquals(x, _x[0]))
            return 0;
        if (_numOps.GreaterThanOrEquals(x, _x[_x.Length - 1]))
            return _x.Length - 2;

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