namespace AiDotNet.Regression;

public class PrincipalComponentRegression<T> : RegressionBase<T>
{
    private readonly PrincipalComponentRegressionOptions<T> _options;
    private Matrix<T> _components;
    private Vector<T> _xMean;
    private Vector<T> _yMean;
    private Vector<T> _xStd;
    private T _yStd;

    public PrincipalComponentRegression(PrincipalComponentRegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new PrincipalComponentRegressionOptions<T>();
        _components = new Matrix<T>(0, 0, NumOps);
        _xMean = Vector<T>.Empty();
        _yMean = Vector<T>.Empty();
        _xStd = Vector<T>.Empty();
        _yStd = NumOps.Zero;
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputs(x, y);

        // Center and scale the data
        (Matrix<T> xScaled, Vector<T> yScaled, _xMean, _xStd, _yStd) = RegressionHelper<T>.CenterAndScale(x, y);

        // Perform PCA
        (Matrix<T> components, Vector<T> explainedVariance) = PerformPCA(xScaled);

        // Select number of components
        int numComponents = SelectNumberOfComponents(explainedVariance);
        _components = components.Submatrix(0, 0, components.Rows, numComponents);

        // Project data onto principal components
        Matrix<T> xProjected = xScaled.Multiply(_components);

        // Perform linear regression on projected data
        Vector<T> coefficients = SolveSystem(xProjected, yScaled);

        // Transform coefficients back to original space
        Coefficients = _components.Multiply(coefficients);

        // Apply regularization to coefficients
        Coefficients = Regularization.RegularizeCoefficients(Coefficients);

        // Adjust for scaling
        for (int i = 0; i < Coefficients.Length; i++)
        {
            Coefficients[i] = NumOps.Divide(NumOps.Multiply(Coefficients[i], _yStd), _xStd[i]);
        }

        // Calculate intercept
        Intercept = NumOps.Subtract(_yMean[0], Coefficients.DotProduct(_xMean));
    }

    private void ValidateInputs(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of rows in x must match the length of y.");
        }
    }

    private (Matrix<T>, Vector<T>) PerformPCA(Matrix<T> x)
    {
        // Perform SVD
        var svd = new SvdDecomposition<T>(x);
        (Matrix<T> u, Vector<T> s, Matrix<T> vt) = (svd.U, svd.S, svd.Vt);

        // Components are the right singular vectors (rows of vt)
        Matrix<T> components = vt.Transpose();

        // Calculate explained variance
        Vector<T> explainedVariance = s.Transform(val => NumOps.Multiply(val, val));
        T totalVariance = explainedVariance.Sum();
        explainedVariance = explainedVariance.Transform(val => NumOps.Divide(val, totalVariance));

        return (components, explainedVariance);
    }

    private int SelectNumberOfComponents(Vector<T> explainedVariance)
    {
        if (_options.NumComponents > 0)
        {
            return Math.Min(_options.NumComponents, explainedVariance.Length);
        }

        T cumulativeVariance = NumOps.Zero;
        for (int i = 0; i < explainedVariance.Length; i++)
        {
            cumulativeVariance = NumOps.Add(cumulativeVariance, explainedVariance[i]);
            if (NumOps.GreaterThanOrEquals(cumulativeVariance, NumOps.FromDouble(_options.ExplainedVarianceRatio)))
            {
                return i + 1;
            }
        }

        return explainedVariance.Length;
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        // Scale the input
        Matrix<T> scaledInput = new Matrix<T>(input.Rows, input.Columns, NumOps);
        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < input.Columns; j++)
            {
                scaledInput[i, j] = NumOps.Divide(NumOps.Subtract(input[i, j], _xMean[j]), _xStd[j]);
            }
        }

        // Make predictions
        Vector<T> predictions = scaledInput.Multiply(Coefficients);
        for (int i = 0; i < predictions.Length; i++)
        {
            predictions[i] = NumOps.Add(NumOps.Multiply(predictions[i], _yStd), _yMean[0]);
        }

        return predictions;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = GetModelType(),
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Coefficients", Coefficients },
                { "Components", _components },
                { "NumComponents", _components.Columns },
                { "FeatureImportance", CalculateFeatureImportances() }
            }
        };
    }

    protected override ModelType GetModelType() => ModelType.PrincipalComponentRegression;

    protected override Vector<T> CalculateFeatureImportances()
    {
        // Feature importances are based on the magnitude of the coefficients
        return Coefficients.Transform(NumOps.Abs);
    }

    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Write base class data
        base.Serialize();

        // Write PCR-specific data
        writer.Write(_options.NumComponents);
        writer.Write(_options.ExplainedVarianceRatio);
        SerializationHelper<T>.SerializeMatrix(writer, _components);
        SerializationHelper<T>.SerializeVector(writer, _xMean);
        SerializationHelper<T>.SerializeVector(writer, _yMean);
        SerializationHelper<T>.SerializeVector(writer, _xStd);
        SerializationHelper<T>.WriteValue(writer, _yStd);

        return ms.ToArray();
    }

    public override void Deserialize(byte[] modelData)
    {
        using MemoryStream ms = new MemoryStream(modelData);
        using BinaryReader reader = new BinaryReader(ms);

        // Read base class data
        base.Deserialize(modelData);

        // Read PCR-specific data
        _options.NumComponents = reader.ReadInt32();
        _options.ExplainedVarianceRatio = reader.ReadDouble();
        _components = SerializationHelper<T>.DeserializeMatrix(reader);
        _xMean = SerializationHelper<T>.DeserializeVector(reader);
        _yMean = SerializationHelper<T>.DeserializeVector(reader);
        _xStd = SerializationHelper<T>.DeserializeVector(reader);
        _yStd = SerializationHelper<T>.ReadValue(reader);
    }
}