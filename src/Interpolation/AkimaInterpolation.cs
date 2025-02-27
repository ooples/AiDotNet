namespace AiDotNet.Interpolation;

public class AkimaInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _b;
    private readonly Vector<T> _c;
    private readonly Vector<T> _d;
    private readonly INumericOperations<T> _numOps;

    public AkimaInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        if (x.Length < 5)
        {
            throw new ArgumentException("Akima interpolation requires at least 5 points.");
        }

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();

        int n = x.Length;
        _b = new Vector<T>(n - 1);
        _c = new Vector<T>(n - 1);
        _d = new Vector<T>(n - 1);

        CalculateCoefficients();
    }

    public T Interpolate(T x)
    {
        int i = FindInterval(x);

        T dx = _numOps.Subtract(x, _x[i]);
        T y = _y[i];
        y = _numOps.Add(y, _numOps.Multiply(_b[i], dx));
        y = _numOps.Add(y, _numOps.Multiply(_c[i], _numOps.Multiply(dx, dx)));
        y = _numOps.Add(y, _numOps.Multiply(_d[i], _numOps.Multiply(_numOps.Multiply(dx, dx), dx)));

        return y;
    }

    private void CalculateCoefficients()
    {
        int n = _x.Length;
        Vector<T> m = new Vector<T>(n + 3);

        // Calculate slopes
        for (int i = 0; i < n - 1; i++)
        {
            m[i + 2] = _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), _numOps.Subtract(_x[i + 1], _x[i]));
        }

        // Set up the first and last two slopes
        m[1] = _numOps.Add(m[2], _numOps.Subtract(m[2], m[3]));
        m[0] = _numOps.Add(m[1], _numOps.Subtract(m[1], m[2]));
        m[n + 1] = _numOps.Add(m[n], _numOps.Subtract(m[n], m[n - 1]));
        m[n + 2] = _numOps.Add(m[n + 1], _numOps.Subtract(m[n + 1], m[n]));

        // Calculate Akima weights
        for (int i = 0; i < n - 1; i++)
        {
            T w1 = _numOps.Abs(_numOps.Subtract(m[i + 3], m[i + 2]));
            T w2 = _numOps.Abs(_numOps.Subtract(m[i + 1], m[i]));

            if (_numOps.Equals(w1, _numOps.Zero) && _numOps.Equals(w2, _numOps.Zero))
            {
                _b[i] = m[i + 2];
            }
            else
            {
                _b[i] = _numOps.Divide(_numOps.Add(_numOps.Multiply(w1, m[i + 1]), _numOps.Multiply(w2, m[i + 2])), _numOps.Add(w1, w2));
            }
        }

        // Calculate remaining coefficients
        for (int i = 0; i < n - 1; i++)
        {
            T h = _numOps.Subtract(_x[i + 1], _x[i]);
            T slope = _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), h);
            _c[i] = _numOps.Divide(_numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(3), slope), _numOps.Add(_numOps.Multiply(_numOps.FromDouble(2), _b[i]), _b[i + 1])), h);
            _d[i] = _numOps.Divide(_numOps.Add(_numOps.Subtract(_b[i], slope), _numOps.Subtract(slope, _b[i + 1])), _numOps.Multiply(h, h));
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