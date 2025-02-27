namespace AiDotNet.GaussianProcesses;

public class SparseGaussianProcess<T> : IGaussianProcess<T>
{
    private IKernelFunction<T> _kernel;
    private Matrix<T> _X;
    private Vector<T> _y;
    private Matrix<T> _inducingPoints;
    private INumericOperations<T> _numOps;
    private Matrix<T> _L;
    private Matrix<T> _V;
    private Vector<T> _D;
    private Vector<T> _alpha;
    private readonly MatrixDecompositionType _decompositionType;

    public SparseGaussianProcess(IKernelFunction<T> kernel, MatrixDecompositionType decompositionType = MatrixDecompositionType.Cholesky)
    {
        _kernel = kernel;
        _X = Matrix<T>.Empty();
        _y = Vector<T>.Empty();
        _L = Matrix<T>.Empty();
        _V = Matrix<T>.Empty();
        _D = Vector<T>.Empty();
        _alpha = Vector<T>.Empty();
        _inducingPoints = Matrix<T>.Empty();
        _decompositionType = decompositionType;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public void Fit(Matrix<T> X, Vector<T> y)
    {
        _X = X;
        _y = y;
        _inducingPoints = SelectInducingPoints(X);

        // Sparse GP training algorithm (Fully Independent Training Conditional - FITC)
        var Kuu = CalculateKernelMatrix(_inducingPoints, _inducingPoints);
        var Kuf = CalculateKernelMatrix(_inducingPoints, X);
        var Kff_diag = CalculateKernelDiagonal(X);

        var choleskyKuu = new CholeskyDecomposition<T>(Kuu);
        var L = choleskyKuu.L;

        // Solve for each column of Kuf separately
        var V = new Matrix<T>(Kuu.Rows, Kuf.Columns);
        for (int i = 0; i < Kuf.Columns; i++)
        {
            var column = Kuf.GetColumn(i);
            var solvedColumn = choleskyKuu.Solve(column);
            V.SetColumn(i, solvedColumn);
        }

        // Calculate Qff_diag
        var Qff_diag = new Vector<T>(Kuf.Columns);
        for (int i = 0; i < Kuf.Columns; i++)
        {
            var column = V.GetColumn(i);
            Qff_diag[i] = _numOps.Square(column.Sum());
        }

        var Lambda = Kff_diag.Subtract(Qff_diag);

        var noise = _numOps.FromDouble(1e-6); // Small noise term for numerical stability
        var D = Lambda.Add(noise).Transform(v => Reciprocal(v));

        var Ky = Kuu.Add(Kuf.Multiply(D.CreateDiagonal()).Multiply(Kuf.Transpose()));
        var choleskyKy = new CholeskyDecomposition<T>(Ky);
        var alpha = choleskyKy.Solve(Kuf.Multiply(D.CreateDiagonal()).Multiply(y));

        // Store necessary components for prediction
        _L = L;
        _V = V;
        _D = D;
        _alpha = alpha;
    }

    private T Reciprocal(T value)
    {
        return _numOps.Divide(_numOps.One, value);
    }

    public (T mean, T variance) Predict(Vector<T> x)
    {
        var Kus = CalculateKernelVector(_inducingPoints, x);
        var Kss = _kernel.Calculate(x, x);

        var f_mean = Kus.DotProduct(_alpha);

        var v = MatrixSolutionHelper.SolveLinearSystem(_L, Kus, _decompositionType);
        var f_var = _numOps.Subtract(Kss, v.DotProduct(v));

        return (f_mean, f_var);
    }

    public void UpdateKernel(IKernelFunction<T> kernel)
    {
        _kernel = kernel;
        if (!_X.IsEmpty && !_y.IsEmpty)
        {
            Fit(_X, _y);
        }
    }

    private Matrix<T> SelectInducingPoints(Matrix<T> X)
    {
        int m = Math.Min(X.Rows, 100); // Number of inducing points, capped at 100 or the number of data points
        var indices = new List<int>();
        var random = new Random();

        while (indices.Count < m)
        {
            int index = random.Next(0, X.Rows);
            if (!indices.Contains(index))
            {
                indices.Add(index);
            }
        }

        return X.GetRows(indices);
    }

    private Matrix<T> CalculateKernelMatrix(Matrix<T> X1, Matrix<T> X2)
    {
        var K = new Matrix<T>(X1.Rows, X2.Rows);
        for (int i = 0; i < X1.Rows; i++)
        {
            for (int j = 0; j < X2.Rows; j++)
            {
                K[i, j] = _kernel.Calculate(X1.GetRow(i), X2.GetRow(j));
            }
        }

        return K;
    }

    private Vector<T> CalculateKernelVector(Matrix<T> X, Vector<T> x)
    {
        var k = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            k[i] = _kernel.Calculate(X.GetRow(i), x);
        }

        return k;
    }

    private Vector<T> CalculateKernelDiagonal(Matrix<T> X)
    {
        var diag = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            diag[i] = _kernel.Calculate(X.GetRow(i), X.GetRow(i));
        }

        return diag;
    }
}