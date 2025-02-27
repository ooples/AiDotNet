namespace AiDotNet.Regression;

public class LogisticRegression<T> : RegressionBase<T>
{
    private readonly LogisticRegressionOptions<T> _options;

    public LogisticRegression(LogisticRegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new LogisticRegressionOptions<T>();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
            throw new ArgumentException("The number of rows in X must match the length of y.");

        int n = x.Rows;
        int p = x.Columns;

        Coefficients = new Vector<T>(p);
        Intercept = NumOps.Zero;

        // Apply regularization to the input matrix
        Matrix<T> regularizedX = Regularization != null ? Regularization.RegularizeMatrix(x) : x;

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            Vector<T> predictions = Predict(regularizedX);
            Vector<T> errors = y.Subtract(predictions);
            Vector<T> gradient = regularizedX.Transpose().Multiply(errors);

            // Apply regularization to the gradient
            if (Regularization != null)
            {
                gradient = ApplyRegularizationGradient(gradient);
            }

            Coefficients = Coefficients.Add(gradient.Multiply(NumOps.FromDouble(_options.LearningRate)));
            Intercept = NumOps.Add(Intercept, NumOps.Multiply(NumOps.FromDouble(_options.LearningRate), errors.Sum()));

            if (HasConverged(gradient))
                break;
        }

        // Apply final regularization to coefficients
        if (Regularization != null)
        {
            Coefficients = Regularization.RegularizeCoefficients(Coefficients);
        }
    }

    private Vector<T> ApplyRegularizationGradient(Vector<T> gradient)
    {
        if (Regularization != null)
        {
            return Regularization.RegularizeGradient(gradient, Coefficients);
        }

        return gradient;
    }

    public override Vector<T> Predict(Matrix<T> x)
    {
        Vector<T> scores = x.Multiply(Coefficients).Add(Intercept);
        return scores.Transform(Sigmoid);
    }

    private T Sigmoid(T x)
    {
        T expNegX = NumOps.Exp(NumOps.Negate(x));
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
    }

    private bool HasConverged(Vector<T> gradient)
    {
        T maxGradient = gradient.Max(NumOps.Abs) ?? NumOps.Zero;
        return NumOps.LessThan(maxGradient, NumOps.FromDouble(_options.Tolerance));
    }

    protected override ModelType GetModelType()
    {
        return ModelType.LogisticRegression;
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize MultipleRegression specific data
        writer.Write(_options.MaxIterations);
        writer.Write(_options.Tolerance);

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

        // Deserialize MultipleRegression specific data
        _options.MaxIterations = reader.ReadInt32();
        _options.Tolerance = reader.ReadDouble();
    }
}