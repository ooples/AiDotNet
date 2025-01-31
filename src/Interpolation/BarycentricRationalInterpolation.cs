namespace AiDotNet.Interpolation;

public class BarycentricRationalInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _weights;
    private readonly INumericOperations<T> _numOps;

    public BarycentricRationalInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        if (x.Length < 2)
        {
            throw new ArgumentException("Barycentric rational interpolation requires at least 2 points.");
        }

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
        _weights = CalculateWeights();
    }

    public T Interpolate(T x)
    {
        T numerator = _numOps.Zero;
        T denominator = _numOps.Zero;

        for (int i = 0; i < _x.Length; i++)
        {
            if (_numOps.Equals(x, _x[i]))
            {
                return _y[i]; // Return exact value if x matches a known point
            }

            T diff = _numOps.Subtract(x, _x[i]);
            T term = _numOps.Divide(_weights[i], diff);

            numerator = _numOps.Add(numerator, _numOps.Multiply(term, _y[i]));
            denominator = _numOps.Add(denominator, term);
        }

        return _numOps.Divide(numerator, denominator);
    }

    private Vector<T> CalculateWeights()
    {
        int n = _x.Length;
        Vector<T> weights = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T weight = _numOps.One;
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    weight = _numOps.Multiply(weight, _numOps.Subtract(_x[i], _x[j]));
                }
            }
            weights[i] = _numOps.Divide(_numOps.One, weight);
        }

        return weights;
    }
}