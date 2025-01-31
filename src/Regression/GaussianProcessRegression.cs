
namespace AiDotNet.Regression;

public class GaussianProcessRegression<T> : NonLinearRegressionBase<T>
{
    private Matrix<T> _kernelMatrix;
    private Vector<T> _alpha;
    private new GaussianProcessRegressionOptions Options { get; }

    public GaussianProcessRegression(GaussianProcessRegressionOptions? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        Options = options ?? new GaussianProcessRegressionOptions();
        _kernelMatrix = new Matrix<T>(0, 0);
        _alpha = new Vector<T>(0);
    }

    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        _kernelMatrix = new Matrix<T>(n, n);

        // Compute the kernel matrix
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                T value = KernelFunction(x.GetRow(i), x.GetRow(j));
                _kernelMatrix[i, j] = value;
                _kernelMatrix[j, i] = value;
            }
        }

        // Add noise to the diagonal for numerical stability
        for (int i = 0; i < n; i++)
        {
            _kernelMatrix[i, i] = NumOps.Add(_kernelMatrix[i, i], NumOps.FromDouble(Options.NoiseLevel));
        }

        if (Options.OptimizeHyperparameters)
        {
            OptimizeHyperparameters(x, y);
        }

        // Apply regularization to the kernel matrix
        Matrix<T> regularizedKernelMatrix = Regularization.RegularizeMatrix(_kernelMatrix);

        // Solve (K + σ²I + R)α = y, where R is the regularization term
        _alpha = MatrixSolutionHelper.SolveLinearSystem(regularizedKernelMatrix, y, Options.DecompositionType);

        // Apply regularization to the alpha coefficients
        _alpha = Regularization.RegularizeCoefficients(_alpha);

        // Store x as support vectors for prediction
        SupportVectors = x;
        Alphas = _alpha;
    }

    private void OptimizeHyperparameters(Matrix<T> x, Vector<T> y)
    {
        int maxIterations = Options.MaxIterations;
        double tolerance = Options.Tolerance;
        double learningRate = 0.01;

        double lengthScale = Options.LengthScale;
        double signalVariance = Options.SignalVariance;
        double prevLogLikelihood = double.NegativeInfinity;

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Compute the kernel matrix with current hyperparameters
            Matrix<T> K = ComputeKernelMatrix(x, lengthScale, signalVariance);

            // Add noise to the diagonal for numerical stability
            for (int i = 0; i < K.Rows; i++)
            {
                K[i, i] = NumOps.Add(K[i, i], NumOps.FromDouble(Options.NoiseLevel));
            }

            // Compute the log likelihood
            double logLikelihood = ComputeLogLikelihood(K, y);

            // Check for convergence
            if (Math.Abs(logLikelihood - prevLogLikelihood) < tolerance)
            {
                break;
            }

            // Compute gradients
            (double gradLengthScale, double gradSignalVariance) = ComputeGradients(x, y, K, lengthScale, signalVariance);

            // Update hyperparameters
            lengthScale += learningRate * gradLengthScale;
            signalVariance += learningRate * gradSignalVariance;

            // Ensure hyperparameters remain positive
            lengthScale = Math.Max(lengthScale, 1e-6);
            signalVariance = Math.Max(signalVariance, 1e-6);

            prevLogLikelihood = logLikelihood;
        }

        // Update the options with optimized hyperparameters
        Options.LengthScale = lengthScale;
        Options.SignalVariance = signalVariance;
    }

    private Matrix<T> ComputeKernelMatrix(Matrix<T> x, double lengthScale, double signalVariance)
    {
        int n = x.Rows;
        var K = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                T value = RBFKernel(x.GetRow(i), x.GetRow(j), lengthScale, signalVariance);
                K[i, j] = value;
                K[j, i] = value;
            }
        }

        return K;
    }

    private T RBFKernel(Vector<T> x1, Vector<T> x2, double lengthScale, double signalVariance)
    {
        T squaredDistance = x1.Subtract(x2).PointwiseMultiply(x1.Subtract(x2)).Sum();
        return NumOps.Multiply(NumOps.FromDouble(signalVariance), 
            NumOps.Exp(NumOps.Divide(NumOps.Negate(squaredDistance), NumOps.FromDouble(2 * lengthScale * lengthScale))));
    }

    private double ComputeLogLikelihood(Matrix<T> K, Vector<T> y)
    {
        var choleskyDecomposition = new CholeskyDecomposition<T>(K);
        Vector<T> alpha = choleskyDecomposition.Solve(y);

        double logDeterminant = 0;
        for (int i = 0; i < K.Rows; i++)
        {
            logDeterminant += Math.Log(Convert.ToDouble(K[i, i]));
        }

        return -0.5 * Convert.ToDouble(y.DotProduct(alpha)) - 0.5 * logDeterminant - 0.5 * K.Rows * Math.Log(2 * Math.PI);
    }

    private (double gradLengthScale, double gradSignalVariance) ComputeGradients(Matrix<T> x, Vector<T> y, Matrix<T> K, double lengthScale, double signalVariance)
    {
        var choleskyDecomposition = new CholeskyDecomposition<T>(K);
        Vector<T> alpha = choleskyDecomposition.Solve(y);

        Matrix<T> KInverse = MatrixHelper<T>.InvertUsingDecomposition(choleskyDecomposition);

        Matrix<T> dK_dLengthScale = ComputeKernelMatrixDerivative(x, lengthScale, signalVariance, true);
        Matrix<T> dK_dSignalVariance = ComputeKernelMatrixDerivative(x, lengthScale, signalVariance, false);

        double gradLengthScale = 0.5 * Convert.ToDouble(alpha.DotProduct(dK_dLengthScale.Multiply(alpha))) - 0.5 * Convert.ToDouble(KInverse.ElementWiseMultiplyAndSum(dK_dLengthScale));
        double gradSignalVariance = 0.5 * Convert.ToDouble(alpha.DotProduct(dK_dSignalVariance.Multiply(alpha))) - 0.5 * Convert.ToDouble(KInverse.ElementWiseMultiplyAndSum(dK_dSignalVariance));

        return (gradLengthScale, gradSignalVariance);
    }

    private Matrix<T> ComputeKernelMatrixDerivative(Matrix<T> x, double lengthScale, double signalVariance, bool isLengthScale)
    {
        int n = x.Rows;
        var dK = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                T value = RBFKernelDerivative(x.GetRow(i), x.GetRow(j), lengthScale, signalVariance, isLengthScale);
                dK[i, j] = value;
                dK[j, i] = value;
            }
        }

        return dK;
    }

    private T RBFKernelDerivative(Vector<T> x1, Vector<T> x2, double lengthScale, double signalVariance, bool isLengthScale)
    {
        T squaredDistance = x1.Subtract(x2).PointwiseMultiply(x1.Subtract(x2)).Sum();
        T kernelValue = RBFKernel(x1, x2, lengthScale, signalVariance);

        if (isLengthScale)
        {
            return NumOps.Multiply(kernelValue, NumOps.Divide(squaredDistance, NumOps.FromDouble(lengthScale * lengthScale * lengthScale)));
        }
        else
        {
            return NumOps.Divide(kernelValue, NumOps.FromDouble(signalVariance));
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NoiseLevel"] = Options.NoiseLevel;
        metadata.AdditionalInfo["OptimizeHyperparameters"] = Options.OptimizeHyperparameters;
        metadata.AdditionalInfo["MaxIterations"] = Options.MaxIterations;
        metadata.AdditionalInfo["Tolerance"] = Options.Tolerance;
        metadata.AdditionalInfo["LengthScale"] = Options.LengthScale;
        metadata.AdditionalInfo["SignalVariance"] = Options.SignalVariance;

        return metadata;
    }

    protected override ModelType GetModelType()
    {
        return ModelType.GaussianProcessRegression;
    }
}