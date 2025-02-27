namespace AiDotNet.GaussianProcesses;

public class StandardGaussianProcess<T> : IGaussianProcess<T>
{
    private IKernelFunction<T> _kernel;
    private Matrix<T> _X;
    private Vector<T> _y;
    private Matrix<T> _K;
    private INumericOperations<T> _numOps;
    private readonly MatrixDecompositionType _decompositionType;

    public StandardGaussianProcess(IKernelFunction<T> kernel, MatrixDecompositionType decompositionType = MatrixDecompositionType.Cholesky)
    {
        _kernel = kernel;
        _X = Matrix<T>.Empty();
        _y = Vector<T>.Empty();
        _K = Matrix<T>.Empty();
        _numOps = MathHelper.GetNumericOperations<T>();
        _decompositionType = decompositionType;
    }

    public void Fit(Matrix<T> X, Vector<T> y)
    {
        _X = X;
        _y = y;
        _K = CalculateKernelMatrix(X, X);
    }

    public (T mean, T variance) Predict(Vector<T> x)
    {
        var k = CalculateKernelVector(_X, x);
    
        // Solve _K * alpha = _y
        var alpha = MatrixSolutionHelper.SolveLinearSystem(_K, _y, _decompositionType);
        var mean = k.DotProduct(alpha);
    
        // Solve _K * v = k
        var v = MatrixSolutionHelper.SolveLinearSystem(_K, k, _decompositionType);
        var variance = _numOps.Subtract(_kernel.Calculate(x, x), k.DotProduct(v));

        return (mean, variance);
    }

    public void UpdateKernel(IKernelFunction<T> kernel)
    {
        _kernel = kernel;
        if (_X != null && _y != null)
        {
            Fit(_X, _y);
        }
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
}