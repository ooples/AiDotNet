namespace AiDotNet.Interpolation;

public class AdaptiveCubicSplineInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly IInterpolation<T> _naturalSpline;
    private readonly IInterpolation<T> _monotonicSpline;
    private readonly bool[] _useMonotonic;
    private readonly INumericOperations<T> _numOps;

    public AdaptiveCubicSplineInterpolation(Vector<T> x, Vector<T> y, T threshold)
    {
        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
        _naturalSpline = new CubicSplineInterpolation<T>(x, y);
        _monotonicSpline = new MonotoneCubicInterpolation<T>(x, y);
        _useMonotonic = DetermineInterpolationMethod(threshold);
    }

    public T Interpolate(T x)
    {
        int i = FindInterval(x);
        return _useMonotonic[i] ? _monotonicSpline.Interpolate(x) : _naturalSpline.Interpolate(x);
    }

    private bool[] DetermineInterpolationMethod(T threshold)
    {
        bool[] useMonotonic = new bool[_x.Length - 1];
        
        for (int i = 0; i < _x.Length - 1; i++)
        {
            T slope = _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), _numOps.Subtract(_x[i + 1], _x[i]));
            T naturalValue = _naturalSpline.Interpolate(_numOps.Divide(_numOps.Add(_x[i], _x[i + 1]), _numOps.FromDouble(2)));
            T monotonicValue = _monotonicSpline.Interpolate(_numOps.Divide(_numOps.Add(_x[i], _x[i + 1]), _numOps.FromDouble(2)));
            
            T naturalDiff = _numOps.Abs(_numOps.Subtract(naturalValue, _numOps.Divide(_numOps.Add(_y[i], _y[i + 1]), _numOps.FromDouble(2))));
            T monotonicDiff = _numOps.Abs(_numOps.Subtract(monotonicValue, _numOps.Divide(_numOps.Add(_y[i], _y[i + 1]), _numOps.FromDouble(2))));
            
            useMonotonic[i] = _numOps.GreaterThan(_numOps.Subtract(naturalDiff, monotonicDiff), threshold);
        }
        
        return useMonotonic;
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