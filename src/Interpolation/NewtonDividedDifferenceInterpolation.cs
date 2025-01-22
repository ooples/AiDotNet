namespace AiDotNet.Interpolation;

public class NewtonDividedDifferenceInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _coefficients;
    private readonly INumericOperations<T> _numOps;

    public NewtonDividedDifferenceInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        if (x.Length < 2)
        {
            throw new ArgumentException("Newton's divided difference interpolation requires at least 2 points.");
        }

        _x = x;
        _numOps = MathHelper.GetNumericOperations<T>();
        _coefficients = CalculateCoefficients(x, y);
    }

    public T Interpolate(T x)
    {
        T result = _coefficients[0];
        T term = _numOps.One;

        for (int i = 1; i < _coefficients.Length; i++)
        {
            term = _numOps.Multiply(term, _numOps.Subtract(x, _x[i - 1]));
            result = _numOps.Add(result, _numOps.Multiply(_coefficients[i], term));
        }

        return result;
    }

    private Vector<T> CalculateCoefficients(Vector<T> x, Vector<T> y)
    {
        int n = x.Length;
        Vector<T> coefficients = new Vector<T>(n, _numOps);
        T[,] dividedDifferences = new T[n, n];

        // Initialize the first column with y values
        for (int i = 0; i < n; i++)
        {
            dividedDifferences[i, 0] = y[i];
        }

        // Calculate divided differences
        for (int j = 1; j < n; j++)
        {
            for (int i = 0; i < n - j; i++)
            {
                dividedDifferences[i, j] = _numOps.Divide(
                    _numOps.Subtract(dividedDifferences[i + 1, j - 1], dividedDifferences[i, j - 1]),
                    _numOps.Subtract(x[i + j], x[i])
                );
            }
        }

        // Extract coefficients from the first row of divided differences
        for (int i = 0; i < n; i++)
        {
            coefficients[i] = dividedDifferences[0, i];
        }

        return coefficients;
    }
}