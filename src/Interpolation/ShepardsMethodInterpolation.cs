namespace AiDotNet.Interpolation;

public class ShepardsMethodInterpolation<T> : I2DInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _z;
    private readonly T _power;
    private readonly INumericOperations<T> _numOps;

    public ShepardsMethodInterpolation(Vector<T> x, Vector<T> y, Vector<T> z, double power = 2.0)
    {
        if (x.Length != y.Length || x.Length != z.Length)
            throw new ArgumentException("Input vectors must have the same length.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
        _power = _numOps.FromDouble(power);
    }

    public T Interpolate(T x, T y)
    {
        T numerator = _numOps.Zero;
        T denominator = _numOps.Zero;

        for (int i = 0; i < _x.Length; i++)
        {
            T distance = CalculateDistance(x, y, _x[i], _y[i]);
            
            if (_numOps.Equals(distance, _numOps.Zero))
            {
                return _z[i]; // Return exact value if the point coincides with a known point
            }

            T weight = _numOps.Power(MathHelper.Reciprocal(distance), _power);
            numerator = _numOps.Add(numerator, _numOps.Multiply(weight, _z[i]));
            denominator = _numOps.Add(denominator, weight);
        }

        return _numOps.Divide(numerator, denominator);
    }

    private T CalculateDistance(T x1, T y1, T x2, T y2)
    {
        T dx = _numOps.Subtract(x1, x2);
        T dy = _numOps.Subtract(y1, y2);

        return _numOps.Sqrt(_numOps.Add(_numOps.Multiply(dx, dx), _numOps.Multiply(dy, dy)));
    }
}