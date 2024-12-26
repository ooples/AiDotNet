namespace AiDotNet.Regression;

public class KernelRidgeRegression<T> : NonLinearRegressionBase<T>
{
    private Matrix<T> _gramMatrix;
    private Vector<T> _dualCoefficients;
    private new KernelRidgeRegressionOptions Options => (KernelRidgeRegressionOptions)base.Options;

    public KernelRidgeRegression(KernelRidgeRegressionOptions options, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _gramMatrix = Matrix<T>.Empty();
        _dualCoefficients = Vector<T>.Empty();
    }

    protected override void OptimizeModel(Matrix<T> X, Vector<T> y)
    {
        int n = X.Rows;
        _gramMatrix = new Matrix<T>(n, n, NumOps);

        // Compute the Gram matrix
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                T value = KernelFunction(X.GetRow(i), X.GetRow(j));
                _gramMatrix[i, j] = value;
                _gramMatrix[j, i] = value;
            }
        }

        // Add ridge penalty to the diagonal
        for (int i = 0; i < n; i++)
        {
            _gramMatrix[i, i] = NumOps.Add(_gramMatrix[i, i], NumOps.FromDouble(Options.LambdaKRR));
        }

        // Apply regularization to the Gram matrix
        Matrix<T> regularizedGramMatrix = Regularization.RegularizeMatrix(_gramMatrix);

        // Solve (K + λI + R)α = y, where R is the regularization term
        _dualCoefficients = MatrixSolutionHelper.SolveLinearSystem(regularizedGramMatrix, y, Options.DecompositionType);

        // Apply regularization to the dual coefficients
        _dualCoefficients = Regularization.RegularizeCoefficients(_dualCoefficients);

        // Store X as support vectors for prediction
        SupportVectors = X;
        Alphas = _dualCoefficients;
    }

    protected override T PredictSingle(Vector<T> input)
    {
        T result = NumOps.Zero;
        for (int i = 0; i < SupportVectors.Rows; i++)
        {
            Vector<T> supportVector = SupportVectors.GetRow(i);
            result = NumOps.Add(result, NumOps.Multiply(Alphas[i], KernelFunction(input, supportVector)));
        }
        return result;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["LambdaKRR"] = Options.LambdaKRR;
        metadata.AdditionalInfo["RegularizationType"] = Regularization.GetType().Name;

        return metadata;
    }

    protected override ModelType GetModelType()
    {
        return ModelType.KernelRidgeRegression;
    }
}