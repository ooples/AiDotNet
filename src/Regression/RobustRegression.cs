namespace AiDotNet.Regression;

public class RobustRegression<T> : RegressionBase<T>
{
    private readonly RobustRegressionOptions<T> _options;

    public RobustRegression(RobustRegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new RobustRegressionOptions<T>();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int p = x.Columns;

        // Apply regularization to the input matrix
        x = Regularization.RegularizeMatrix(x);

        // Initial regression estimate
        IRegression<T> initialRegression = _options.InitialRegression ?? new MultipleRegression<T>();
        initialRegression.Train(x, y);
        Coefficients = initialRegression.Coefficients;
        Intercept = initialRegression.Intercept;

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            Vector<T> residuals = y.Subtract(Predict(x));
            Vector<T> weights = WeightFunctionHelper.CalculateWeights(residuals, _options.WeightFunction, _options.TuningConstant, NumOps);

            // Weighted least squares
            Matrix<T> weightedX = x.PointwiseMultiply(weights);
            Vector<T> weightedY = y.PointwiseMultiply(weights);

            var wls = new MultipleRegression<T>();
            wls.Train(weightedX, weightedY);

            Vector<T> newCoefficients = wls.Coefficients;
            T newIntercept = wls.Intercept;

            // Check for convergence
            if (IsConverged(Coefficients, newCoefficients, Intercept, newIntercept))
            {
                break;
            }

            Coefficients = newCoefficients;
            Intercept = newIntercept;
        }

        // Apply regularization to the coefficients
        Coefficients = Regularization.RegularizeCoefficients(Coefficients);
    }

    private bool IsConverged(Vector<T> oldCoefficients, Vector<T> newCoefficients, T oldIntercept, T newIntercept)
    {
        T tolerance = NumOps.FromDouble(_options.Tolerance);
        
        for (int i = 0; i < oldCoefficients.Length; i++)
        {
            if (NumOps.GreaterThan(NumOps.Abs(NumOps.Subtract(oldCoefficients[i], newCoefficients[i])), tolerance))
            {
                return false;
            }
        }

        return NumOps.LessThanOrEquals(NumOps.Abs(NumOps.Subtract(oldIntercept, newIntercept)), tolerance);
    }

    protected override ModelType GetModelType() => ModelType.RobustRegression;

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize RobustRegression specific options
        writer.Write(_options.TuningConstant);
        writer.Write(_options.MaxIterations);
        writer.Write(_options.Tolerance);
        writer.Write((int)_options.WeightFunction);

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

        // Deserialize RobustRegression specific options
        _options.TuningConstant = reader.ReadDouble();
        _options.MaxIterations = reader.ReadInt32();
        _options.Tolerance = reader.ReadDouble();
        _options.WeightFunction = (WeightFunction)reader.ReadInt32();
    }
}