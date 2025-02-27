namespace AiDotNet.Regression;

public class NegativeBinomialRegression<T> : RegressionBase<T>
{
    private T _dispersion;
    private readonly NegativeBinomialRegressionOptions<T> _options;

    public NegativeBinomialRegression(NegativeBinomialRegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new NegativeBinomialRegressionOptions<T>();
        _dispersion = NumOps.One;
    }

    public override void Train(Matrix<T> X, Vector<T> y)
    {
        if (X.Rows != y.Length)
            throw new ArgumentException("The number of rows in X must match the length of y.");

        InitializeCoefficients(X.Columns);

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var oldCoefficients = Coefficients.Copy();

            // Calculate linear predictors and means
            var linearPredictors = X.Multiply(Coefficients).Add(Intercept);
            var means = linearPredictors.Transform(NumOps.Exp);

            // Calculate weights and working response
            var weights = CalculateWeights(means);
            var workingResponse = CalculateWorkingResponse(y, means, linearPredictors);

            // Weighted least squares step
            var weightedX = X.PointwiseMultiply(weights.Transform(NumOps.Sqrt));
            var weightedY = workingResponse.PointwiseMultiply(weights.Transform(NumOps.Sqrt));

            // Apply regularization to the design matrix
            var regularizedX = Regularization.RegularizeMatrix(weightedX);

            // Solve the regularized system
            var newCoefficients = MatrixSolutionHelper.SolveLinearSystem(regularizedX, weightedY, MatrixDecompositionFactory.GetDecompositionType(_options.DecompositionMethod));

            // Apply regularization to the coefficients
            newCoefficients = Regularization.RegularizeCoefficients(newCoefficients);

            Coefficients = newCoefficients.Slice(1, newCoefficients.Length - 1);
            Intercept = newCoefficients[0];

            // Check for convergence
            if (NumOps.LessThan(Coefficients.Subtract(oldCoefficients).Norm(), NumOps.FromDouble(_options.Tolerance)))
                break;
        }

        UpdateDispersion(X, y);
    }

    private void InitializeCoefficients(int featureCount)
    {
        Coefficients = new Vector<T>(featureCount);
        Intercept = NumOps.Zero;
    }

    public override Vector<T> Predict(Matrix<T> X)
    {
        var linearPredictors = X.Multiply(Coefficients).Add(Intercept);
        return linearPredictors.Transform(NumOps.Exp);
    }

    private Vector<T> CalculateWeights(Vector<T> means)
    {
        return means.Transform(mu => NumOps.Divide(NumOps.Square(mu), NumOps.Add(mu, NumOps.Divide(NumOps.Square(mu), _dispersion))));
    }

    private Vector<T> CalculateWorkingResponse(Vector<T> y, Vector<T> means, Vector<T> linearPredictors)
    {
        return linearPredictors.Add(y.Subtract(means).PointwiseDivide(means));
    }

    private void UpdateDispersion(Matrix<T> X, Vector<T> y)
    {
        var predictions = Predict(X);
        var pearsonResiduals = y.Subtract(predictions).Transform(
            (yi, predi) => NumOps.Divide(NumOps.Subtract(yi, NumOps.FromDouble(predi)), NumOps.Sqrt(NumOps.FromDouble(predi))));
        var sumSquaredResiduals = pearsonResiduals.Transform(NumOps.Square).Sum();
        var degreesOfFreedom = NumOps.FromDouble(X.Rows - X.Columns);
        _dispersion = NumOps.Divide(sumSquaredResiduals, degreesOfFreedom);
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize NegativeBinomialRegression specific data
        writer.Write(Convert.ToDouble(_dispersion));
        writer.Write(_options.MaxIterations);
        writer.Write(Convert.ToDouble(_options.Tolerance));

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

        // Deserialize NegativeBinomialRegression specific data
        _dispersion = NumOps.FromDouble(reader.ReadDouble());
        _options.MaxIterations = reader.ReadInt32();
        _options.Tolerance = reader.ReadDouble();
    }

    protected override ModelType GetModelType()
    {
        return ModelType.NegativeBinomialRegression;
    }
}