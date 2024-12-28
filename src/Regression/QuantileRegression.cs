namespace AiDotNet.Regression;

public class QuantileRegression<T> : RegressionBase<T>
{
    private readonly QuantileRegressionOptions<T> _options;

    public QuantileRegression(QuantileRegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new QuantileRegressionOptions<T>();
    }
     
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int p = x.Columns;

        // Initialize coefficients
        Coefficients = new Vector<T>(p, NumOps);
        Intercept = NumOps.Zero;

        // Apply regularization to the input matrix
        x = Regularization.RegularizeMatrix(x);

        // Gradient descent optimization
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            Vector<T> gradients = new(p, NumOps);
            T interceptGradient = NumOps.Zero;

            for (int i = 0; i < n; i++)
            {
                T prediction = Predict(x.GetRow(i));
                T error = NumOps.Subtract(y[i], prediction);
                T gradient = NumOps.GreaterThan(error, NumOps.Zero) 
                    ? NumOps.FromDouble(_options.Quantile) 
                    : NumOps.FromDouble(_options.Quantile - 1);

                for (int j = 0; j < p; j++)
                {
                    gradients[j] = NumOps.Add(gradients[j], NumOps.Multiply(gradient, x[i, j]));
                }
                interceptGradient = NumOps.Add(interceptGradient, gradient);
            }

            // Update coefficients and intercept
            for (int j = 0; j < p; j++)
            {
                Coefficients[j] = NumOps.Add(Coefficients[j], NumOps.Multiply(NumOps.FromDouble(_options.LearningRate), gradients[j]));
            }
            Intercept = NumOps.Add(Intercept, NumOps.Multiply(NumOps.FromDouble(_options.LearningRate), interceptGradient));

            // Apply regularization to coefficients
            Coefficients = Regularization.RegularizeCoefficients(Coefficients);
        }
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows, NumOps);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = Predict(input.GetRow(i));
        }

        return predictions;
    }

    private T Predict(Vector<T> input)
    {
        return NumOps.Add(Coefficients.DotProduct(input), Intercept);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Quantile"] = _options.Quantile;

        return metadata;
    }

    protected override ModelType GetModelType()
    {
        return ModelType.QuantileRegression;
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize QuantileRegression specific data
        writer.Write(_options.Quantile);
        writer.Write(_options.LearningRate);
        writer.Write(_options.MaxIterations);

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

        // Deserialize QuantileRegression specific data
        _options.Quantile = reader.ReadDouble();
        _options.LearningRate = reader.ReadDouble();
        _options.MaxIterations = reader.ReadInt32();
    }
}