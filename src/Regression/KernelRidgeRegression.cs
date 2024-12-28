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

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize KernelRidgeRegression specific data
        writer.Write(Options.LambdaKRR);
        writer.Write((int)Options.DecompositionType);

        // Serialize _gramMatrix
        writer.Write(_gramMatrix.Rows);
        writer.Write(_gramMatrix.Columns);
        for (int i = 0; i < _gramMatrix.Rows; i++)
        {
            for (int j = 0; j < _gramMatrix.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_gramMatrix[i, j]));
            }
        }

        // Serialize _dualCoefficients
        writer.Write(_dualCoefficients.Length);
        for (int i = 0; i < _dualCoefficients.Length; i++)
        {
            writer.Write(Convert.ToDouble(_dualCoefficients[i]));
        }

        return ms.ToArray();
    }

    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize KernelRidgeRegression specific data
        Options.LambdaKRR = reader.ReadDouble();
        Options.DecompositionType = (MatrixDecompositionType)reader.ReadInt32();

        // Deserialize _gramMatrix
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        _gramMatrix = new Matrix<T>(rows, cols, NumOps);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                _gramMatrix[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Deserialize _dualCoefficients
        int length = reader.ReadInt32();
        _dualCoefficients = new Vector<T>(length, NumOps);
        for (int i = 0; i < length; i++)
        {
            _dualCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}