namespace AiDotNet.Interpolation;

public class NaturalSplineInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly int _degree;
    private readonly IMatrixDecomposition<T>? _decomposition;
    private readonly Vector<T>[] _coefficients;
    private readonly INumericOperations<T> _numOps;

    public NaturalSplineInterpolation(Vector<T> x, Vector<T> y, int degree = 3, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 2)
            throw new ArgumentException("Natural spline interpolation requires at least 2 points.");
        if (degree < 2)
            throw new ArgumentException("Degree must be at least 2 for natural spline interpolation.");

        _x = x;
        _y = y;
        _degree = degree;
        _decomposition = decomposition;
        _numOps = MathHelper.GetNumericOperations<T>();
        _coefficients = new Vector<T>[_degree];

        for (int i = 0; i < _degree; i++)
        {
            _coefficients[i] = new Vector<T>(x.Length - 1, _numOps);
        }

        CalculateCoefficients();
    }

    public T Interpolate(T x)
    {
        int i = FindInterval(x);
        T dx = _numOps.Subtract(x, _x[i]);
        T result = _y[i];

        for (int j = 1; j < _degree; j++)
        {
            result = _numOps.Add(result, _numOps.Multiply(_coefficients[j][i], Power(dx, j)));
        }

        return result;
    }

    private void CalculateCoefficients()
    {
        int n = _x.Length;
        Matrix<T> A = new Matrix<T>(n, n, _numOps);
        Vector<T> b = new Vector<T>(n, _numOps);

        // Set up the system of equations
        for (int i = 0; i < n - 1; i++)
        {
            T h = _numOps.Subtract(_x[i + 1], _x[i]);
            A[i, i] = h;
            if (i < n - 2)
                A[i, i + 1] = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Add(h, _numOps.Subtract(_x[i + 2], _x[i + 1])));
            if (i > 0)
                A[i, i - 1] = h;

            if (i < n - 2)
            {
                T dy1 = _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), h);
                T dy2 = _numOps.Divide(_numOps.Subtract(_y[i + 2], _y[i + 1]), _numOps.Subtract(_x[i + 2], _x[i + 1]));
                b[i] = _numOps.Multiply(_numOps.FromDouble(6), _numOps.Subtract(dy2, dy1));
            }
        }

        // Apply natural spline conditions
        A[0, 0] = _numOps.One;
        A[n - 1, n - 1] = _numOps.One;
        b[0] = _numOps.Zero;
        b[n - 1] = _numOps.Zero;

        // Solve the system
        var decomposition = _decomposition ?? new LuDecomposition<T>(A);
        Vector<T> m = MatrixSolutionHelper.SolveLinearSystem(b, decomposition);

        // Calculate the coefficients
        for (int i = 0; i < n - 1; i++)
        {
            T h = _numOps.Subtract(_x[i + 1], _x[i]);
            _coefficients[0][i] = _y[i];
            _coefficients[1][i] = _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), h);
            _coefficients[1][i] = _numOps.Subtract(_coefficients[1][i], _numOps.Multiply(_numOps.Divide(h, _numOps.FromDouble(6)), _numOps.Add(_numOps.Multiply(_numOps.FromDouble(2), m[i]), m[i + 1])));
            _coefficients[2][i] = _numOps.Divide(m[i], _numOps.FromDouble(2));
            _coefficients[3][i] = _numOps.Divide(_numOps.Subtract(m[i + 1], m[i]), _numOps.Multiply(_numOps.FromDouble(6), h));
        }
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

    private T Power(T x, int power)
    {
        T result = _numOps.One;
        for (int i = 0; i < power; i++)
        {
            result = _numOps.Multiply(result, x);
        }

        return result;
    }
}