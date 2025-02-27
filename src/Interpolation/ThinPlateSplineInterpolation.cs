namespace AiDotNet.Interpolation;

public class ThinPlateSplineInterpolation<T> : I2DInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _z;
    private Vector<T> _weights;
    private T _a0;
    private T _ax;
    private T _ay;
    private readonly INumericOperations<T> _numOps;
    private readonly IMatrixDecomposition<T>? _decomposition;

    public ThinPlateSplineInterpolation(Vector<T> x, Vector<T> y, Vector<T> z, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length || x.Length != z.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 3)
            throw new ArgumentException("At least 3 points are required for Thin Plate Spline interpolation.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
        _a0 = _numOps.Zero;
        _ax = _numOps.Zero;
        _ay = _numOps.Zero;
        _weights = Vector<T>.Empty();
        _decomposition = decomposition;

        CalculateWeights();
    }

    public T Interpolate(T x, T y)
    {
        T result = _numOps.Add(
            _numOps.Add(_a0, _numOps.Multiply(_ax, x)),
            _numOps.Multiply(_ay, y)
        );

        for (int i = 0; i < _x.Length; i++)
        {
            T r = CalculateDistance(x, y, _x[i], _y[i]);
            result = _numOps.Add(result, _numOps.Multiply(_weights[i], RadialBasisFunction(r)));
        }

        return result;
    }

    private void CalculateWeights()
    {
        int n = _x.Length;
        Matrix<T> L = new Matrix<T>(n + 3, n + 3);
        Vector<T> rhs = new Vector<T>(n + 3);

        // Fill the L matrix
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                T r = CalculateDistance(_x[i], _y[i], _x[j], _y[j]);
                T value = RadialBasisFunction(r);
                L[i, j] = value;
                L[j, i] = value;
            }
            L[i, n] = _numOps.One;
            L[i, n + 1] = _x[i];
            L[i, n + 2] = _y[i];
            L[n, i] = _numOps.One;
            L[n + 1, i] = _x[i];
            L[n + 2, i] = _y[i];
            rhs[i] = _z[i];
        }

        // Solve the system
        var decomposition = _decomposition ?? new LuDecomposition<T>(L);
        Vector<T> solution = MatrixSolutionHelper.SolveLinearSystem(rhs, decomposition);

        // Extract weights and coefficients
        _weights = solution.GetSubVector(0, n);
        _a0 = solution[n];
        _ax = solution[n + 1];
        _ay = solution[n + 2];
    }

    private T CalculateDistance(T x1, T y1, T x2, T y2)
    {
        T dx = _numOps.Subtract(x1, x2);
        T dy = _numOps.Subtract(y1, y2);

        return _numOps.Sqrt(_numOps.Add(_numOps.Multiply(dx, dx), _numOps.Multiply(dy, dy)));
    }

    private T RadialBasisFunction(T r)
    {
        if (_numOps.Equals(r, _numOps.Zero))
            return _numOps.Zero;
        
        return _numOps.Multiply(
            _numOps.Multiply(r, r),
            _numOps.Log(r)
        );
    }
}