namespace AiDotNet.Regression;

public class PartialLeastSquaresRegression<T> : RegressionBase<T>
{
    private readonly PartialLeastSquaresRegressionOptions<T> _options;
    private Matrix<T> _loadings;
    private Matrix<T> _scores;
    private Matrix<T> _weights;
    private Vector<T> _yMean;
    private Vector<T> _xMean;
    private T _yStd;
    private Vector<T> _xStd;

    public PartialLeastSquaresRegression(PartialLeastSquaresRegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new PartialLeastSquaresRegressionOptions<T>();
        _loadings = new Matrix<T>(0, 0);
        _scores = new Matrix<T>(0, 0);
        _weights = new Matrix<T>(0, 0);
        _yMean = new Vector<T>(0);
        _xMean = new Vector<T>(0);
        _yStd = NumOps.Zero;
        _xStd = new Vector<T>(0);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputs(x, y);
    
        // Center and scale the data
        (Matrix<T> xScaled, Vector<T> yScaled, _xMean, _xStd, _yStd) = RegressionHelper<T>.CenterAndScale(x, y);

        int numComponents = Math.Min(_options.NumComponents, x.Columns);
        _loadings = new Matrix<T>(x.Columns, numComponents);
        _scores = new Matrix<T>(x.Rows, numComponents);
        _weights = new Matrix<T>(x.Columns, numComponents);

        Matrix<T> xResidual = xScaled.Copy();
        Vector<T> yResidual = yScaled.Copy();

        for (int i = 0; i < numComponents; i++)
        {
            Vector<T> w = xResidual.Transpose().Multiply(yResidual);
            w = w.Normalize();

            Vector<T> t = xResidual.Multiply(w);
            T tt = t.DotProduct(t);

            Vector<T> p = xResidual.Transpose().Multiply(t).Divide(tt);
            T q = NumOps.Divide(yResidual.DotProduct(t), tt);

            xResidual = xResidual.Subtract(t.OuterProduct(p));
            yResidual = yResidual.Subtract(t.Multiply(q));

            _loadings.SetColumn(i, p);
            _scores.SetColumn(i, t);
            _weights.SetColumn(i, w);
        }

        // Calculate regression coefficients
        Matrix<T> W = _weights;
        Matrix<T> P = _loadings;
        Matrix<T> invPtW = (P.Transpose().Multiply(W)).Inverse();
        Coefficients = W.Multiply(invPtW).Multiply(_scores.Transpose()).Multiply(yScaled);

        // Apply regularization to coefficients
        Coefficients = Regularization.RegularizeCoefficients(Coefficients);

        // Adjust for scaling
        for (int i = 0; i < Coefficients.Length; i++)
        {
            Coefficients[i] = NumOps.Divide(NumOps.Multiply(Coefficients[i], _yStd), _xStd[i]);
        }

        // Calculate intercept
        Intercept = NumOps.Subtract(_yMean[0], Coefficients.DotProduct(_xMean));

        // Apply regularization to the model matrices
        _loadings = Regularization.RegularizeMatrix(_loadings);
        _scores = Regularization.RegularizeMatrix(_scores);
        _weights = Regularization.RegularizeMatrix(_weights);
    }

    private void ValidateInputs(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of rows in x must match the length of y.");
        }
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        int rows = input.Rows;
        int cols = input.Columns;

        // Scale the input
        Matrix<T> scaledInput = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                scaledInput[i, j] = NumOps.Divide(NumOps.Subtract(input[i, j], _xMean[j]), _xStd[j]);
            }
        }
    
        // Make predictions
        Vector<T> predictions = scaledInput.Multiply(Coefficients);
        for (int i = 0; i < predictions.Length; i++)
        {
            predictions[i] = NumOps.Add(predictions[i], Intercept);
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
                { "Loadings", _loadings },
                { "Scores", _scores },
                { "Weights", _weights },
                { "NumComponents", _options.NumComponents },
                { "FeatureImportance", CalculateFeatureImportances() }
            }
        };
    }

    protected override ModelType GetModelType() => ModelType.PartialLeastSquaresRegression;

    protected override Vector<T> CalculateFeatureImportances()
    {
        // VIP (Variable Importance in Projection) scores
        Vector<T> vip = new Vector<T>(Coefficients.Length);
    
        // Calculate ssY (sum of squares of Y)
        T ssY = NumOps.Zero;
        Matrix<T> scoresTransposeMultiplyScores = _scores.Transpose().Multiply(_scores);
        for (int i = 0; i < scoresTransposeMultiplyScores.Rows; i++)
        {
            ssY = NumOps.Add(ssY, scoresTransposeMultiplyScores[i, i]);
        }

        for (int j = 0; j < Coefficients.Length; j++)
        {
            T score = NumOps.Zero;
            for (int a = 0; a < _options.NumComponents; a++)
            {
                T w = _weights[j, a];
                T t = _scores.GetColumn(a).DotProduct(_scores.GetColumn(a));
                score = NumOps.Add(score, NumOps.Multiply(NumOps.Multiply(w, w), t));
            }
            vip[j] = NumOps.Multiply(NumOps.Sqrt(NumOps.Multiply(NumOps.FromDouble(Coefficients.Length), NumOps.Divide(score, ssY))), Coefficients[j]);
        }

        return vip;
    }

    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Write base class data
        base.Serialize();

        // Write PLS-specific data
        writer.Write(_options.NumComponents);
        SerializationHelper<T>.SerializeMatrix(writer, _loadings);
        SerializationHelper<T>.SerializeMatrix(writer, _scores);
        SerializationHelper<T>.SerializeMatrix(writer, _weights);
        SerializationHelper<T>.SerializeVector(writer, _yMean);
        SerializationHelper<T>.SerializeVector(writer, _xMean);
        SerializationHelper<T>.WriteValue(writer, _yStd);
        SerializationHelper<T>.SerializeVector(writer, _xStd);

        return ms.ToArray();
    }

    public override void Deserialize(byte[] modelData)
    {
        using MemoryStream ms = new MemoryStream(modelData);
        using BinaryReader reader = new BinaryReader(ms);

        // Read base class data
        base.Deserialize(modelData);

        // Read PLS-specific data
        _options.NumComponents = reader.ReadInt32();
        _loadings = SerializationHelper<T>.DeserializeMatrix(reader);
        _scores = SerializationHelper<T>.DeserializeMatrix(reader);
        _weights = SerializationHelper<T>.DeserializeMatrix(reader);
        _yMean = SerializationHelper<T>.DeserializeVector(reader);
        _xMean = SerializationHelper<T>.DeserializeVector(reader);
        _yStd = SerializationHelper<T>.ReadValue(reader);
        _xStd = SerializationHelper<T>.DeserializeVector(reader);
    }
}