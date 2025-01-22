namespace AiDotNet.Interpolation;

public class MultiquadricInterpolation<T> : I2DInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _z;
    private readonly T _epsilon;
    private readonly INumericOperations<T> _numOps;
    private readonly IMatrixDecomposition<T>? _decomposition;
    private Vector<T>? _coefficients;

    public MultiquadricInterpolation(Vector<T> x, Vector<T> y, Vector<T> z, double epsilon = 1.0, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length || x.Length != z.Length)
            throw new ArgumentException("Input vectors must have the same length.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
        _decomposition = decomposition;
        CalculateCoefficients();
    }

    private void CalculateCoefficients()
    {
        int n = _x.Length;
        var A = new Matrix<T>(n, n, _numOps);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T dx = _numOps.Subtract(_x[i], _x[j]);
                T dy = _numOps.Subtract(_y[i], _y[j]);
                T r = _numOps.Sqrt(_numOps.Add(_numOps.Multiply(dx, dx), _numOps.Multiply(dy, dy)));
                A[i, j] = MultiquadricBasis(r);
            }
        }

        var decomposition = _decomposition ?? new LuDecomposition<T>(A);
        _coefficients = MatrixSolutionHelper.SolveLinearSystem(_z, decomposition);
    }

    public T Interpolate(T x, T y)
    {
        if (_coefficients == null)
            throw new InvalidOperationException("Coefficients have not been calculated.");

        T result = _numOps.Zero;
        for (int i = 0; i < _x.Length; i++)
        {
            T dx = _numOps.Subtract(x, _x[i]);
            T dy = _numOps.Subtract(y, _y[i]);
            T r = _numOps.Sqrt(_numOps.Add(_numOps.Multiply(dx, dx), _numOps.Multiply(dy, dy)));
            result = _numOps.Add(result, _numOps.Multiply(_coefficients[i], MultiquadricBasis(r)));
        }

        return result;
    }

    private T MultiquadricBasis(T r)
    {
        return _numOps.Sqrt(_numOps.Add(_numOps.Multiply(r, r), _numOps.Multiply(_epsilon, _epsilon)));
    }
}