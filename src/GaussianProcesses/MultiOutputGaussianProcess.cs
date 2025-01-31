namespace AiDotNet.GaussianProcesses;

public class MultiOutputGaussianProcess<T> : IGaussianProcess<T>
{
    private IKernelFunction<T> _kernel;
    private Matrix<T> _X;
    private Matrix<T> _Y;
    private Matrix<T> _K;
    private Matrix<T> _L;
    private Matrix<T> _alpha;
    private INumericOperations<T> _numOps;

    public MultiOutputGaussianProcess(IKernelFunction<T> kernel)
    {
        _kernel = kernel;
        _numOps = MathHelper.GetNumericOperations<T>();
        _X = Matrix<T>.Empty();
        _Y = Matrix<T>.Empty();
        _K = Matrix<T>.Empty();
        _L = Matrix<T>.Empty();
        _alpha = Matrix<T>.Empty();
    }

    public void Fit(Matrix<T> X, Vector<T> y)
    {
        throw new InvalidOperationException("Use FitMultiOutput method for multi-output GP");
    }

    public void FitMultiOutput(Matrix<T> X, Matrix<T> Y)
    {
        _X = X;
        _Y = Y;
        
        // Calculate the kernel matrix
        _K = CalculateKernelMatrix(X, X);
        
        // Add a small value to the diagonal for numerical stability
        for (int i = 0; i < _K.Rows; i++)
        {
            _K[i, i] = _numOps.Add(_K[i, i], _numOps.FromDouble(1e-8));
        }

        // Solve for alpha using Cholesky decomposition
        _alpha = new Matrix<T>(Y.Rows, Y.Columns);
        for (int i = 0; i < Y.Columns; i++)
        {
            var y_col = Y.GetColumn(i);
            var alpha_col = MatrixSolutionHelper.SolveLinearSystem(_K, y_col, MatrixDecompositionType.Cholesky);
            for (int j = 0; j < Y.Rows; j++)
            {
                _alpha[j, i] = alpha_col[j];
            }
        }

        // Store the Cholesky decomposition for later use in predictions
        _L = new CholeskyDecomposition<T>(_K).L;
    }

    public (T mean, T variance) Predict(Vector<T> x)
    {
        throw new InvalidOperationException("Use PredictMultiOutput method for multi-output GP");
    }

    public (Vector<T> means, Matrix<T> covariance) PredictMultiOutput(Vector<T> x)
    {
        var k_star = CalculateKernelVector(_X, x);
        var means = new Vector<T>(_Y.Columns);

        for (int i = 0; i < _Y.Columns; i++)
        {
            means[i] = _numOps.Zero;
            for (int j = 0; j < k_star.Length; j++)
            {
                means[i] = _numOps.Add(means[i], _numOps.Multiply(k_star[j], _alpha[j, i]));
            }
        }

        var v = MatrixSolutionHelper.SolveLinearSystem(_K, k_star, MatrixDecompositionType.Cholesky);
        var variance = _numOps.Subtract(_kernel.Calculate(x, x), k_star.DotProduct(v));
        var covariance = new Matrix<T>(_Y.Columns, _Y.Columns);

        for (int i = 0; i < _Y.Columns; i++)
        {
            covariance[i, i] = variance;
        }

        return (means, covariance);
    }

    public void UpdateKernel(IKernelFunction<T> kernel)
    {
        _kernel = kernel;
        if (!_X.IsEmpty && !_Y.IsEmpty)
        {
            FitMultiOutput(_X, _Y);
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