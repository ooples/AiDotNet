namespace AiDotNet.Interpolation;

public class WhittakerShannonInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly INumericOperations<T> _numOps;

    public WhittakerShannonInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();

        if (!IsUniformlySampled())
        {
            Console.WriteLine("Warning: Input data is not uniformly sampled. Interpolation results may be less accurate.");
        }
    }

    public T Interpolate(T x)
    {
        T result = _numOps.Zero;

        for (int i = 0; i < _x.Length; i++)
        {
            T t = _numOps.Subtract(x, _x[i]);
            T sincValue = Sinc(t);
            result = _numOps.Add(result, _numOps.Multiply(_y[i], sincValue));
        }

        return result;
    }

    private T Sinc(T x)
    {
        if (_numOps.Equals(x, _numOps.Zero))
            return _numOps.One;

        T piX = _numOps.Multiply(MathHelper.Pi<T>(), x);
        return _numOps.Divide(MathHelper.Sin(piX), piX);
    }

    private bool IsUniformlySampled()
    {
        if (_x.Length <= 2) return true;

        T interval = _numOps.Subtract(_x[1], _x[0]);
        T tolerance = _numOps.Multiply(interval, _numOps.FromDouble(1e-6)); // Adjust tolerance as needed

        for (int i = 2; i < _x.Length; i++)
        {
            T currentInterval = _numOps.Subtract(_x[i], _x[i-1]);
            if (_numOps.GreaterThan(_numOps.Abs(_numOps.Subtract(currentInterval, interval)), tolerance))
            {
                return false;
            }
        }

        return true;
    }
}