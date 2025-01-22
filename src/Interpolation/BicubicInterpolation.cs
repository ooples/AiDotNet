namespace AiDotNet.Interpolation;

public class BicubicInterpolation<T> : I2DInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Matrix<T> _z;
    private readonly INumericOperations<T> _numOps;
    private readonly IMatrixDecomposition<T>? _decomposition;

    public BicubicInterpolation(Vector<T> x, Vector<T> y, Matrix<T> z, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != z.Rows || y.Length != z.Columns)
            throw new ArgumentException("Input dimensions mismatch.");
        if (x.Length < 4 || y.Length < 4)
            throw new ArgumentException("Bicubic interpolation requires at least a 4x4 grid.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
        _decomposition = decomposition;
    }

    public T Interpolate(T x, T y)
    {
        int i = FindInterval(_x, x);
        int j = FindInterval(_y, y);

        T[,] p = new T[4, 4];
        for (int m = -1; m <= 2; m++)
        {
            for (int n = -1; n <= 2; n++)
            {
                p[m + 1, n + 1] = _z[MathHelper.Clamp(i + m, 0, _z.Rows - 1), MathHelper.Clamp(j + n, 0, _z.Columns - 1)];
            }
        }

        T dx = _numOps.Divide(_numOps.Subtract(x, _x[i]), _numOps.Subtract(_x[i + 1], _x[i]));
        T dy = _numOps.Divide(_numOps.Subtract(y, _y[j]), _numOps.Subtract(_y[j + 1], _y[j]));

        return InterpolateBicubicPatch(p, dx, dy);
    }

    private T InterpolateBicubicPatch(T[,] p, T x, T y)
    {
        T[,] a = CalculateBicubicCoefficients(p);
        T result = _numOps.Zero;

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result = _numOps.Add(result, _numOps.Multiply(a[i, j], 
                    _numOps.Multiply(_numOps.Power(x, _numOps.FromDouble(i)), _numOps.Power(y, _numOps.FromDouble(j)))));
            }
        }

        return result;
    }

    private T[,] CalculateBicubicCoefficients(T[,] p)
    {
        T[,] a = new T[4, 4];
        Matrix<T> coefficients = new Matrix<T>(16, 16, _numOps);
        Vector<T> values = new Vector<T>(16, _numOps);

        // Fill the coefficients matrix and values vector
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                int index = i * 4 + j;
                values[index] = p[i, j];

                for (int k = 0; k < 4; k++)
                {
                    for (int l = 0; l < 4; l++)
                    {
                        T xTerm = _numOps.Power(_numOps.FromDouble(i - 1), _numOps.FromDouble(k));
                        T yTerm = _numOps.Power(_numOps.FromDouble(j - 1), _numOps.FromDouble(l));
                        coefficients[index, k * 4 + l] = _numOps.Multiply(xTerm, yTerm);
                    }
                }
            }
        }

        // Solve the system
        var decomposition = _decomposition ?? new LuDecomposition<T>(coefficients);
        Vector<T> solution = MatrixSolutionHelper.SolveLinearSystem(values, decomposition);

        // Convert the solution vector to the 4x4 coefficient matrix
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                a[i, j] = solution[i * 4 + j];
            }
        }

        return a;
    }

    private int FindInterval(Vector<T> values, T point)
    {
        for (int i = 0; i < values.Length - 1; i++)
        {
            if (_numOps.LessThanOrEquals(point, values[i + 1]))
                return i;
        }

        return values.Length - 2;
    }
}