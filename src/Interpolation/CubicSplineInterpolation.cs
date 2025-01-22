namespace AiDotNet.Interpolation;

public class CubicSplineInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _a;
    private readonly Vector<T> _b;
    private readonly Vector<T> _c;
    private readonly Vector<T> _d;
    private readonly INumericOperations<T> _numOps;

    public CubicSplineInterpolation(Vector<T> x, Vector<T> y)
    {
        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();

        int n = x.Length - 1;
        _a = y.Copy();
        _b = new Vector<T>(n, _numOps);
        _c = new Vector<T>(n + 1, _numOps);
        _d = new Vector<T>(n, _numOps);

        CalculateCoefficients();
    }

    public T Interpolate(T x)
    {
        int i = FindInterval(x);
        T dx = _numOps.Subtract(x, _x[i]);

        return _numOps.Add(
            _numOps.Add(
                _numOps.Add(
                    _a[i],
                    _numOps.Multiply(_b[i], dx)
                ),
                _numOps.Multiply(_c[i], _numOps.Multiply(dx, dx))
            ),
            _numOps.Multiply(_d[i], _numOps.Multiply(_numOps.Multiply(dx, dx), dx))
        );
    }

    private void CalculateCoefficients()
    {
        int n = _x.Length - 1;
        Vector<T> h = new Vector<T>(n, _numOps);

        for (int i = 0; i < n; i++)
        {
            h[i] = _numOps.Subtract(_x[i + 1], _x[i]);
        }

        Vector<T> alpha = new Vector<T>(n, _numOps);
        for (int i = 1; i < n; i++)
        {
            alpha[i] = _numOps.Multiply(
                _numOps.FromDouble(3),
                _numOps.Subtract(
                    _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), h[i]),
                    _numOps.Divide(_numOps.Subtract(_y[i], _y[i - 1]), h[i - 1])
                )
            );
        }

        Vector<T> l = new Vector<T>(n + 1, _numOps);
        Vector<T> mu = new Vector<T>(n + 1, _numOps);
        Vector<T> z = new Vector<T>(n + 1, _numOps);

        l[0] = _numOps.One;
        mu[0] = _numOps.Zero;
        z[0] = _numOps.Zero;

        for (int i = 1; i < n; i++)
        {
            l[i] = _numOps.Subtract(
                _numOps.Multiply(_numOps.FromDouble(2), _numOps.Add(_x[i + 1], _x[i - 1])),
                _numOps.Multiply(mu[i - 1], h[i - 1])
            );
            mu[i] = _numOps.Divide(h[i], l[i]);
            z[i] = _numOps.Divide(
                _numOps.Subtract(alpha[i], _numOps.Multiply(z[i - 1], h[i - 1])),
                l[i]
            );
        }

        l[n] = _numOps.One;
        z[n] = _numOps.Zero;
        _c[n] = _numOps.Zero;

        for (int j = n - 1; j >= 0; j--)
        {
            _c[j] = _numOps.Subtract(z[j], _numOps.Multiply(mu[j], _c[j + 1]));
            _b[j] = _numOps.Divide(
                _numOps.Subtract(_numOps.Subtract(_y[j + 1], _y[j]), _numOps.Multiply(h[j], _numOps.Add(_c[j + 1], _numOps.Multiply(_numOps.FromDouble(2), _c[j])))),
                h[j]
            );
            _d[j] = _numOps.Divide(_numOps.Subtract(_c[j + 1], _c[j]), h[j]);
        }
    }

    private int FindInterval(T x)
    {
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