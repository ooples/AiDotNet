namespace AiDotNet.Regression;

public class OrthogonalRegression<T> : RegressionBase<T>
{
    private readonly OrthogonalRegressionOptions<T> _options;

    public OrthogonalRegression(OrthogonalRegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new OrthogonalRegressionOptions<T>();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int p = x.Columns;

        // Apply regularization to the input matrix
        x = Regularization.RegularizeMatrix(x);

        // Center the data
        Vector<T> meanX = new(p, NumOps);
        for (int j = 0; j < p; j++)
        {
            meanX[j] = x.GetColumn(j).Mean();
        }
        T meanY = y.Mean();

        Matrix<T> centeredX = new(n, p, NumOps);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                centeredX[i, j] = NumOps.Subtract(x[i, j], meanX[j]);
            }
        }
        Vector<T> centeredY = y.Subtract(meanY);

        // Scale the variables if option is set
        Vector<T> scaleX = Vector<T>.CreateDefault(p, NumOps.One);
        if (_options.ScaleVariables)
        {
            for (int j = 0; j < p; j++)
            {
                T columnVariance = centeredX.GetColumn(j).Variance();
                scaleX[j] = NumOps.Sqrt(columnVariance);
                if (!NumOps.Equals(scaleX[j], NumOps.Zero))
                {
                    for (int i = 0; i < n; i++)
                    {
                        centeredX[i, j] = NumOps.Divide(centeredX[i, j], scaleX[j]);
                    }
                }
            }
        }

        // Compute the augmented matrix [X y]
        Matrix<T> augmentedMatrix = new(n, p + 1, NumOps);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                augmentedMatrix[i, j] = centeredX[i, j];
            }
            augmentedMatrix[i, p] = centeredY[i];
        }

        // Use the decomposition method from options
        IMatrixDecomposition<T> decomposition = Options.DecompositionMethod ?? new SvdDecomposition<T>(augmentedMatrix);

        // Get the solution using MatrixSolutionHelper
        Vector<T> solution = MatrixSolutionHelper.SolveLinearSystem(augmentedMatrix, augmentedMatrix.GetColumn(p), MatrixDecompositionFactory.GetDecompositionType(Options.DecompositionMethod));

        // Rescale the solution
        for (int j = 0; j < p; j++)
        {
            solution[j] = NumOps.Divide(solution[j], scaleX[j]);
        }

        // Normalize the solution
        T norm = NumOps.Sqrt(solution.DotProduct(solution));
        solution = solution.Divide(norm);

        // Extract coefficients and intercept
        Coefficients = solution.GetSubVector(0, p);

        // Apply regularization to the coefficients
        Coefficients = Regularization.RegularizeCoefficients(Coefficients);

        Intercept = NumOps.Subtract(meanY, Coefficients.DotProduct(meanX));
    }

    protected override ModelType GetModelType() => ModelType.OrthogonalRegression;

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize OrthogonalRegression specific options
        writer.Write(_options.Tolerance);
        writer.Write(_options.MaxIterations);
        writer.Write(_options.ScaleVariables);

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

        // Deserialize OrthogonalRegression specific options
        _options.Tolerance = reader.ReadDouble();
        _options.MaxIterations = reader.ReadInt32();
        _options.ScaleVariables = reader.ReadBoolean();
    }
}