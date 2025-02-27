namespace AiDotNet.Interpolation;

public class SincInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly T _cutoffFrequency;
    private readonly INumericOperations<T> _numOps;

    public SincInterpolation(IEnumerable<double> x, IEnumerable<double> y, double cutoffFrequency = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        var xList = x.ToList();
        var yList = y.ToList();

        if (xList.Count != yList.Count)
            throw new ArgumentException("Input vectors must have the same length.");

        _x = new Vector<T>(xList.Select(val => _numOps.FromDouble(val)));
        _y = new Vector<T>(yList.Select(val => _numOps.FromDouble(val)));
        _cutoffFrequency = _numOps.FromDouble(cutoffFrequency);
    }

    public T Interpolate(T x)
    {
        T result = _numOps.Zero;

        for (int i = 0; i < _x.Length; i++)
        {
            T diff = _numOps.Subtract(x, _x[i]);
            T sincValue = Sinc(_numOps.Multiply(_cutoffFrequency, diff));
            result = _numOps.Add(result, _numOps.Multiply(_y[i], sincValue));
        }

        return result;
    }

    private T Sinc(T x)
    {
        if (_numOps.Equals(x, _numOps.Zero))
        {
            return _numOps.One;
        }

        T piX = _numOps.Multiply(MathHelper.Pi<T>(), x);
        return _numOps.Divide(MathHelper.Sin(piX), piX);
    }
}