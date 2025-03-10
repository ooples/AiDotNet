namespace AiDotNet.Interpolation;

public class ClampedSplineInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly T _startSlope;
    private readonly T _endSlope;
    private readonly IMatrixDecomposition<T>? _decomposition;
    private readonly Vector<T>[] _coefficients;
    private readonly INumericOperations<T> _numOps;

    public ClampedSplineInterpolation(Vector<T> x, Vector<T> y, double startSlope = 0.0, double endSlope = 0.0, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 2)
            throw new ArgumentException("Clamped spline interpolation requires at least 2 points.");

        _x = x;
        _y = y;
        _decomposition = decomposition;
        _numOps = MathHelper.GetNumericOperations<T>();

        _startSlope = _numOps.FromDouble(startSlope);
        _endSlope = _numOps.FromDouble(endSlope);

        _coefficients = new Vector<T>[4];

        for (int i = 0; i < 4; i++)
        {
            _coefficients[i] = new Vector<T>(x.Length - 1);
        }

        CalculateCoefficients();
    }

    public T Interpolate(T x)
    {
        int i = FindInterval(x);
        T dx = _numOps.Subtract(x, _x[i]);
        T result = _y[i];

        for (int j = 1; j < 4; j++)
        {
            result = _numOps.Add(result, _numOps.Multiply(_coefficients[j][i], Power(dx, j)));
        }

        return result;
    }

    private void CalculateCoefficients()
    {
        int n = _x.Length;
        Matrix<T> A = new Matrix<T>(n, n);
        Vector<T> b = new Vector<T>(n);

        // Set up the system of equations
        for (int i = 1; i < n - 1; i++)
        {
            T h_prev = _numOps.Subtract(_x[i], _x[i - 1]);
            T h_next = _numOps.Subtract(_x[i + 1], _x[i]);

            A[i, i - 1] = h_prev;
            A[i, i] = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Add(h_prev, h_next));
            A[i, i + 1] = h_next;

            T dy_prev = _numOps.Divide(_numOps.Subtract(_y[i], _y[i - 1]), h_prev);
            T dy_next = _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), h_next);
            b[i] = _numOps.Multiply(_numOps.FromDouble(6), _numOps.Subtract(dy_next, dy_prev));
        }

        // Apply clamped spline conditions
        T h_first = _numOps.Subtract(_x[1], _x[0]);
        T h_last = _numOps.Subtract(_x[n - 1], _x[n - 2]);

        A[0, 0] = _numOps.Multiply(_numOps.FromDouble(2), h_first);
        A[0, 1] = h_first;
        b[0] = _numOps.Multiply(_numOps.FromDouble(6), _numOps.Subtract(_numOps.Divide(_numOps.Subtract(_y[1], _y[0]), h_first), _startSlope));

        A[n - 1, n - 2] = h_last;
        A[n - 1, n - 1] = _numOps.Multiply(_numOps.FromDouble(2), h_last);
        b[n - 1] = _numOps.Multiply(_numOps.FromDouble(6), _numOps.Subtract(_endSlope, _numOps.Divide(_numOps.Subtract(_y[n - 1], _y[n - 2]), h_last)));

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