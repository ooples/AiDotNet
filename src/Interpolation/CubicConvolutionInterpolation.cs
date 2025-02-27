namespace AiDotNet.Interpolation;

public class CubicConvolutionInterpolation<T> : I2DInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Matrix<T> _z;
    private readonly INumericOperations<T> _numOps;

    public CubicConvolutionInterpolation(Vector<T> x, Vector<T> y, Matrix<T> z)
    {
        if (x.Length != z.Rows || y.Length != z.Columns)
            throw new ArgumentException("Input dimensions must match the z matrix dimensions.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Interpolate(T x, T y)
    {
        int i = FindInterval(_x, x);
        int j = FindInterval(_y, y);

        T dx = _numOps.Divide(_numOps.Subtract(x, _x[i]), _numOps.Subtract(_x[i + 1], _x[i]));
        T dy = _numOps.Divide(_numOps.Subtract(y, _y[j]), _numOps.Subtract(_y[j + 1], _y[j]));

        T[,] p = new T[4, 4];
        for (int m = -1; m <= 2; m++)
        {
            for (int n = -1; n <= 2; n++)
            {
                int row = MathHelper.Clamp(i + m, 0, _x.Length - 1);
                int col = MathHelper.Clamp(j + n, 0, _y.Length - 1);
                p[m + 1, n + 1] = _z[row, col];
            }
        }

        return BicubicInterpolate(p, dx, dy);
    }

    private T BicubicInterpolate(T[,] p, T x, T y)
    {
        T[] a = new T[4];
        for (int i = 0; i < 4; i++)
        {
            a[i] = CubicInterpolate(p[i, 0], p[i, 1], p[i, 2], p[i, 3], y);
        }

        return CubicInterpolate(a[0], a[1], a[2], a[3], x);
    }

    private T CubicInterpolate(T p0, T p1, T p2, T p3, T x)
    {
        T a = _numOps.Subtract(_numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(-0.5), p0), _numOps.Multiply(_numOps.FromDouble(1.5), p1)), _numOps.Multiply(_numOps.FromDouble(-1.5), p2));
        a = _numOps.Add(a, _numOps.Multiply(_numOps.FromDouble(0.5), p3));

        T b = _numOps.Add(_numOps.Add(_numOps.Multiply(p0, _numOps.FromDouble(2.5)), _numOps.Multiply(p1, _numOps.FromDouble(-4.5))), _numOps.Multiply(p2, _numOps.FromDouble(3.0)));
        b = _numOps.Subtract(b, _numOps.Multiply(p3, _numOps.FromDouble(0.5)));

        T c = _numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(-0.5), p0), _numOps.Multiply(_numOps.FromDouble(0.5), p2));

        T d = p1;

        return _numOps.Add(_numOps.Add(_numOps.Add(_numOps.Multiply(_numOps.Multiply(a, x), _numOps.Multiply(x, x)), _numOps.Multiply(_numOps.Multiply(b, x), x)), _numOps.Multiply(c, x)), d);
    }

    private int FindInterval(Vector<T> values, T point)
    {
        int index = values.BinarySearch(point);
        if (index < 0)
        {
            index = ~index - 1;
        }

        return MathHelper.Clamp(index, 0, values.Length - 2);
    }
}