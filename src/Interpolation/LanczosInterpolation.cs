namespace AiDotNet.Interpolation;

public class LanczosInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly int _a;
    private readonly INumericOperations<T> _numOps;

    public LanczosInterpolation(Vector<T> x, Vector<T> y, int a = 3)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (a < 1)
            throw new ArgumentException("Parameter 'a' must be a positive integer.");

        _x = x;
        _y = y;
        _a = a;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Interpolate(T x)
    {
        T result = _numOps.Zero;
        T sum = _numOps.Zero;

        for (int i = 0; i < _x.Length; i++)
        {
            T diff = _numOps.Subtract(x, _x[i]);
            T lanczosValue = LanczosKernel(diff);
            result = _numOps.Add(result, _numOps.Multiply(_y[i], lanczosValue));
            sum = _numOps.Add(sum, lanczosValue);
        }

        // Normalize the result
        return _numOps.Divide(result, sum);
    }

    private T LanczosKernel(T x)
    {
        if (_numOps.Equals(x, _numOps.Zero))
        {
            return _numOps.One;
        }

        if (_numOps.GreaterThanOrEquals(_numOps.Abs(x), _numOps.FromDouble(_a)))
        {
            return _numOps.Zero;
        }

        T piX = _numOps.Multiply(MathHelper.Pi<T>(), x);
        T sinc = _numOps.Divide(MathHelper.Sin(piX), piX);
        T sinc2 = _numOps.Divide(MathHelper.Sin(_numOps.Divide(piX, _numOps.FromDouble(_a))), _numOps.Divide(piX, _numOps.FromDouble(_a)));

        return _numOps.Multiply(sinc, sinc2);
    }
}