namespace AiDotNet.Interpolation;

public class PchipInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _slopes;
    private readonly INumericOperations<T> _numOps;

    public PchipInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 2)
            throw new ArgumentException("PCHIP interpolation requires at least 2 points.");

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
        _slopes = new Vector<T>(x.Length);

        CalculateSlopes();
    }

    public T Interpolate(T x)
    {
        int i = FindInterval(x);
        T h = _numOps.Subtract(_x[i + 1], _x[i]);
        T t = _numOps.Divide(_numOps.Subtract(x, _x[i]), h);

        T h00 = _numOps.Add(_numOps.Multiply(_numOps.FromDouble(2), _numOps.Power(t, _numOps.FromDouble(3))), _numOps.Multiply(_numOps.FromDouble(-3), _numOps.Power(t, _numOps.FromDouble(2))));
        h00 = _numOps.Add(h00, _numOps.One);

        T h10 = _numOps.Add(_numOps.Power(t, _numOps.FromDouble(3)), _numOps.Multiply(_numOps.FromDouble(-2), _numOps.Power(t, _numOps.FromDouble(2))));
        h10 = _numOps.Add(h10, t);

        T h01 = _numOps.Add(_numOps.Multiply(_numOps.FromDouble(-2), _numOps.Power(t, _numOps.FromDouble(3))), _numOps.Multiply(_numOps.FromDouble(3), _numOps.Power(t, _numOps.FromDouble(2))));

        T h11 = _numOps.Subtract(_numOps.Power(t, _numOps.FromDouble(3)), _numOps.Power(t, _numOps.FromDouble(2)));

        T result = _numOps.Add(_numOps.Multiply(h00, _y[i]), _numOps.Multiply(h10, _numOps.Multiply(h, _slopes[i])));
        result = _numOps.Add(result, _numOps.Multiply(h01, _y[i + 1]));
        result = _numOps.Add(result, _numOps.Multiply(h11, _numOps.Multiply(h, _slopes[i + 1])));

        return result;
    }

    private void CalculateSlopes()
    {
        int n = _x.Length;

        for (int i = 0; i < n - 1; i++)
        {
            T dx = _numOps.Subtract(_x[i + 1], _x[i]);
            T dy = _numOps.Subtract(_y[i + 1], _y[i]);
            T slope = _numOps.Divide(dy, dx);

            if (i == 0)
            {
                _slopes[i] = slope;
            }
            else if (i == n - 2)
            {
                _slopes[n - 1] = slope;
            }
            else
            {
                T dx_prev = _numOps.Subtract(_x[i], _x[i - 1]);
                T dy_prev = _numOps.Subtract(_y[i], _y[i - 1]);
                T slope_prev = _numOps.Divide(dy_prev, dx_prev);

                _slopes[i] = WeightedHarmonicMean(slope_prev, slope);
            }
        }

        // Adjust slopes to ensure monotonicity
        for (int i = 0; i < n - 1; i++)
        {
            T dx = _numOps.Subtract(_x[i + 1], _x[i]);
            T dy = _numOps.Subtract(_y[i + 1], _y[i]);
            T slope = _numOps.Divide(dy, dx);

            if (_numOps.Equals(slope, _numOps.Zero))
            {
                _slopes[i] = _numOps.Zero;
                _slopes[i + 1] = _numOps.Zero;
            }
            else
            {
                T alpha = _numOps.Divide(_slopes[i], slope);
                T beta = _numOps.Divide(_slopes[i + 1], slope);

                if (_numOps.GreaterThan(_numOps.Add(_numOps.Power(alpha, _numOps.FromDouble(2)), _numOps.Power(beta, _numOps.FromDouble(2))), _numOps.FromDouble(9)))
                {
                    T tau = _numOps.Divide(_numOps.FromDouble(3), _numOps.Sqrt(_numOps.Add(_numOps.Power(alpha, _numOps.FromDouble(2)), _numOps.Power(beta, _numOps.FromDouble(2)))));
                    _slopes[i] = _numOps.Multiply(tau, _numOps.Multiply(alpha, slope));
                    _slopes[i + 1] = _numOps.Multiply(tau, _numOps.Multiply(beta, slope));
                }
            }
        }
    }

    private T WeightedHarmonicMean(T a, T b)
    {
        if (_numOps.Equals(a, _numOps.Zero) || _numOps.Equals(b, _numOps.Zero))
            return _numOps.Zero;

        T w1 = _numOps.Abs(b);
        T w2 = _numOps.Abs(a);

        return _numOps.Divide(_numOps.Add(_numOps.Multiply(w1, a), _numOps.Multiply(w2, b)),
                               _numOps.Add(w1, w2));
    }

    private int FindInterval(T x)
    {
        for (int i = 0; i < _x.Length - 1; i++)
        {
            if (_numOps.LessThanOrEquals(x, _x[i + 1]))
                return i;
        }

        return _x.Length - 2;
    }
}