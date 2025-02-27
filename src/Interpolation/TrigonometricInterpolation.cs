namespace AiDotNet.Interpolation;

public class TrigonometricInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _a;
    private readonly Vector<T> _b;
    private readonly T _period;
    private readonly INumericOperations<T> _numOps;

    public TrigonometricInterpolation(IEnumerable<double> x, IEnumerable<double> y, double? customPeriod = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        var xList = x.ToList();
        var yList = y.ToList();

        if (xList.Count != yList.Count)
            throw new ArgumentException("Input vectors must have the same length.");
        if (xList.Count % 2 == 0)
            throw new ArgumentException("Number of points must be odd for trigonometric interpolation.");

        _x = new Vector<T>(xList.Select(val => _numOps.FromDouble(val)));
        _y = new Vector<T>(yList.Select(val => _numOps.FromDouble(val)));

        // Calculate or set the period
        if (customPeriod.HasValue)
        {
            _period = _numOps.FromDouble(customPeriod.Value);
        }
        else
        {
            double calculatedPeriod = xList.Max() - xList.Min();
            _period = _numOps.FromDouble(calculatedPeriod);
        }

        int n = (xList.Count - 1) / 2;
        _a = new Vector<T>(n + 1);
        _b = new Vector<T>(n);

        CalculateCoefficients();
    }

    public T Interpolate(T x)
    {
        T result = _a[0];
        int n = _a.Length - 1;

        for (int k = 1; k <= n; k++)
        {
            T angle = _numOps.Multiply(_numOps.Divide(_numOps.FromDouble(2 * k * Math.PI), _period), x);
            result = _numOps.Add(result, _numOps.Multiply(_a[k], MathHelper.Cos(angle)));
            
            if (k < n)
            {
                result = _numOps.Add(result, _numOps.Multiply(_b[k - 1], MathHelper.Sin(angle)));
            }
        }

        return result;
    }

    private void CalculateCoefficients()
    {
        int n = (_x.Length - 1) / 2;
        T twoOverN = _numOps.Divide(_numOps.FromDouble(2), _numOps.FromDouble(_x.Length));

        for (int k = 0; k <= n; k++)
        {
            _a[k] = _numOps.Zero;
            for (int j = 0; j < _x.Length; j++)
            {
                T angle = _numOps.Multiply(_numOps.Divide(_numOps.FromDouble(2 * k * Math.PI), _period), _x[j]);
                _a[k] = _numOps.Add(_a[k], _numOps.Multiply(_y[j], MathHelper.Cos(angle)));
            }

            _a[k] = _numOps.Multiply(_a[k], twoOverN);
        }

        for (int k = 1; k < n; k++)
        {
            _b[k - 1] = _numOps.Zero;
            for (int j = 0; j < _x.Length; j++)
            {
                T angle = _numOps.Multiply(_numOps.Divide(_numOps.FromDouble(2 * k * Math.PI), _period), _x[j]);
                _b[k - 1] = _numOps.Add(_b[k - 1], _numOps.Multiply(_y[j], MathHelper.Sin(angle)));
            }

            _b[k - 1] = _numOps.Multiply(_b[k - 1], twoOverN);
        }
    }
}