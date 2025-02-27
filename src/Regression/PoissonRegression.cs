namespace AiDotNet.Regression;

public class PoissonRegression<T> : RegressionBase<T>
{
    private readonly PoissonRegressionOptions<T> _options;

    public PoissonRegression(PoissonRegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new PoissonRegressionOptions<T>();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidationHelper<T>.ValidateInputData(x, y);
        ValidationHelper<T>.ValidatePoissonData(y);

        int numFeatures = x.Columns;
        Coefficients = new Vector<T>(numFeatures);
        Intercept = NumOps.Zero;

        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            Vector<T> currentCoefficients = new([.. Coefficients, Intercept]);
            Vector<T> mu = PredictMean(xWithIntercept, currentCoefficients);
            Matrix<T> w = ComputeWeights(mu);
            Vector<T> z = ComputeWorkingResponse(xWithIntercept, y, mu, currentCoefficients);

            Matrix<T> xTw = xWithIntercept.Transpose().Multiply(w);
            Matrix<T> xTwx = xTw.Multiply(xWithIntercept);
            Vector<T> xTwz = xTw.Multiply(z);

            // Apply regularization to the matrix
            if (Regularization != null)
            {
                xTwx = Regularization.RegularizeMatrix(xTwx);
            }

            Vector<T> newCoefficients = MatrixSolutionHelper.SolveLinearSystem(xTwx, xTwz, MatrixDecompositionFactory.GetDecompositionType(_options.DecompositionMethod));

            // Apply regularization to the coefficients
            if (Regularization != null)
            {
                newCoefficients = Regularization.RegularizeCoefficients(newCoefficients);
            }

            if (HasConverged(currentCoefficients, newCoefficients))
            {
                break;
            }

            Coefficients = new Vector<T>([.. newCoefficients.Take(numFeatures)]);
            Intercept = newCoefficients[numFeatures];
        }
    }

    private Vector<T> PredictMean(Matrix<T> x, Vector<T> coefficients)
    {
        return x.Multiply(coefficients).Transform(NumOps.Exp);
    }

    private Matrix<T> ComputeWeights(Vector<T> mu)
    {
        return Matrix<T>.CreateDiagonal(mu);
    }

    private static Vector<T> ComputeWorkingResponse(Matrix<T> x, Vector<T> y, Vector<T> mu, Vector<T> coefficients)
    {
        Vector<T> eta = x.Multiply(coefficients);
        return eta.Add(y.Subtract(mu).PointwiseDivide(mu));
    }

    private bool HasConverged(Vector<T> oldCoefficients, Vector<T> newCoefficients)
    {
        T diff = oldCoefficients.Subtract(newCoefficients).L2Norm();
        return NumOps.LessThan(diff, NumOps.FromDouble(_options.Tolerance));
    }

    public override Vector<T> Predict(Matrix<T> x)
    {
        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));
        Vector<T> coefficientsWithIntercept = new(Coefficients.Length + 1);

        for (int i = 0; i < Coefficients.Length; i++)
        {
            coefficientsWithIntercept[i] = Coefficients[i];
        }
        coefficientsWithIntercept[Coefficients.Length] = Intercept;

        return PredictMean(xWithIntercept, coefficientsWithIntercept);
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize PoissonRegression specific options
        writer.Write(_options.MaxIterations);
        writer.Write(Convert.ToDouble(_options.Tolerance));

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize PoissonRegression specific options
        _options.MaxIterations = reader.ReadInt32();
        _options.Tolerance = Convert.ToDouble(reader.ReadDouble());
    }

    protected override ModelType GetModelType()
    {
        return ModelType.PoissonRegression;
    }
}