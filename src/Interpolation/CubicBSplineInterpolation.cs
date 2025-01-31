namespace AiDotNet.Interpolation;

public class CubicBSplineInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _knots;
    private readonly Vector<T> _coefficients;
    private readonly INumericOperations<T> _numOps;
    private readonly int _degree;
    private readonly MatrixDecompositionType _decompositionType;

    public CubicBSplineInterpolation(Vector<T> x, Vector<T> y, int degree = 3, MatrixDecompositionType decompositionType = MatrixDecompositionType.Lu)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 4)
            throw new ArgumentException("Cubic B-Spline interpolation requires at least 4 points.");

        _x = x;
        _y = y;
        _degree = degree;
        _decompositionType = decompositionType;
        _numOps = MathHelper.GetNumericOperations<T>();
        _knots = GenerateKnots();
        _coefficients = CalculateCoefficients();
    }

    public T Interpolate(T x)
    {
        if (_numOps.LessThan(x, _x[0]) || _numOps.GreaterThan(x, _x[_x.Length - 1]))
            throw new ArgumentOutOfRangeException(nameof(x), "Interpolation point is outside the data range.");

        int i = FindSpan(x);
        Vector<T> basis = CalculateBasisFunctions(x, i);

        T result = _numOps.Zero;
        for (int j = 0; j < 4; j++)
        {
            result = _numOps.Add(result, _numOps.Multiply(basis[j], _coefficients[i - 3 + j]));
        }

        return result;
    }

    private Vector<T> GenerateKnots()
    {
        int n = _x.Length;
        Vector<T> knots = new Vector<T>(n + 6);

        // Not-a-knot condition
        knots[0] = _numOps.Subtract(_x[0], _numOps.Multiply(_numOps.FromDouble(3), _numOps.Subtract(_x[1], _x[0])));
        knots[1] = _numOps.Subtract(_x[0], _numOps.Multiply(_numOps.FromDouble(2), _numOps.Subtract(_x[1], _x[0])));
        knots[2] = _numOps.Subtract(_x[0], _numOps.Subtract(_x[1], _x[0]));

        for (int i = 0; i < n; i++)
        {
            knots[i + 3] = _x[i];
        }

        // Not-a-knot condition
        knots[n + 3] = _numOps.Add(_x[n - 1], _numOps.Subtract(_x[n - 1], _x[n - 2]));
        knots[n + 4] = _numOps.Add(_x[n - 1], _numOps.Multiply(_numOps.FromDouble(2), _numOps.Subtract(_x[n - 1], _x[n - 2])));
        knots[n + 5] = _numOps.Add(_x[n - 1], _numOps.Multiply(_numOps.FromDouble(3), _numOps.Subtract(_x[n - 1], _x[n - 2])));

        return knots;
    }

    private Vector<T> CalculateCoefficients()
    {
        int n = _x.Length;
        Matrix<T> A = new Matrix<T>(n, n);
        Vector<T> b = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            Vector<T> basis = CalculateBasisFunctions(_x[i], FindSpan(_x[i]));
            for (int j = 0; j < 4; j++)
            {
                if (i - 3 + j >= 0 && i - 3 + j < n)
                {
                    A[i, i - 3 + j] = basis[j];
                }
            }
            b[i] = _y[i];
        }

        return MatrixSolutionHelper.SolveLinearSystem(A, b, _decompositionType);
    }

    private int FindSpan(T x)
    {
        if (_numOps.GreaterThanOrEquals(x, _knots[_knots.Length - 4]))
            return _knots.Length - 5;

        int low = 3;
        int high = _knots.Length - 4;

        while (low < high)
        {
            int mid = (low + high) / 2;
            if (_numOps.LessThan(x, _knots[mid]))
                high = mid;
            else
                low = mid + 1;
        }

        return low - 1;
    }

    private Vector<T> CalculateBasisFunctions(T x, int i)
    {
        Vector<T> basis = new Vector<T>(4);
        Vector<T> left = new Vector<T>(4);
        Vector<T> right = new Vector<T>(4);

        basis[0] = _numOps.One;

        for (int j = 1; j <= 3; j++)
        {
            left[j] = _numOps.Subtract(x, _knots[i + 1 - j]);
            right[j] = _numOps.Subtract(_knots[i + j], x);
            T saved = _numOps.Zero;

            for (int r = 0; r < j; r++)
            {
                T temp = _numOps.Divide(basis[r], _numOps.Add(right[r + 1], left[j - r]));
                basis[r] = _numOps.Add(saved, _numOps.Multiply(right[r + 1], temp));
                saved = _numOps.Multiply(left[j - r], temp);
            }

            basis[j] = saved;
        }

        return basis;
    }
}