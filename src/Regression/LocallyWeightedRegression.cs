namespace AiDotNet.Regression;

public class LocallyWeightedRegression<T> : NonLinearRegressionBase<T>
{
    private readonly LocallyWeightedRegressionOptions _options;
    private Matrix<T> _xTrain;
    private Vector<T> _yTrain;

    public LocallyWeightedRegression(LocallyWeightedRegressionOptions? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new LocallyWeightedRegressionOptions();
        _xTrain = Matrix<T>.Empty();
        _yTrain = Vector<T>.Empty();
    }

    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // In LWR, we don't pre-compute a global model. Instead, we store the training data.
        _xTrain = x;
        _yTrain = y;
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }

        return predictions;
    }

    protected override T PredictSingle(Vector<T> input)
    {
        // Compute weights for each training point
        var weights = ComputeWeights(input);

        // Create the weighted design matrix and target vector
        var weightedX = _xTrain.PointwiseMultiply(weights.CreateDiagonal());
        var weightedY = _yTrain.PointwiseMultiply(weights);

        // Add regularization
        weightedX = Regularization.RegularizeMatrix(weightedX);

        // Solve the weighted least squares problem
        var xTx = weightedX.Transpose().Multiply(weightedX);
        var xTy = weightedX.Transpose().Multiply(weightedY);
        var coefficients = MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _options.DecompositionType);

        // Apply regularization to coefficients
        coefficients = Regularization.RegularizeCoefficients(coefficients);

        // Make prediction
        return input.DotProduct(coefficients);
    }

    private Vector<T> ComputeWeights(Vector<T> input)
    {
        var weights = new Vector<T>(_xTrain.Rows);
        var bandwidth = NumOps.FromDouble(_options.Bandwidth);

        for (int i = 0; i < _xTrain.Rows; i++)
        {
            var distance = EuclideanDistance(input, _xTrain.GetRow(i));
            weights[i] = KernelFunction(NumOps.Divide(distance, bandwidth));
        }

        return weights;
    }

    private T EuclideanDistance(Vector<T> v1, Vector<T> v2)
    {
        return NumOps.Sqrt(v1.Subtract(v2).PointwiseMultiply(v1.Subtract(v2)).Sum());
    }

    private T KernelFunction(T u)
    {
        // Tricube kernel function
        var absU = NumOps.Abs(u);
        if (NumOps.GreaterThan(absU, NumOps.One))
            return NumOps.Zero;
        var temp = NumOps.Subtract(NumOps.One, NumOps.Power(absU, NumOps.FromDouble(3)));
        return NumOps.Power(temp, NumOps.FromDouble(3));
    }

    protected override ModelType GetModelType() => ModelType.LocallyWeightedRegression;

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize LWR specific data
        writer.Write(_options.Bandwidth);
    
        // Serialize _xTrain
        writer.Write(_xTrain.Rows);
        writer.Write(_xTrain.Columns);
        for (int i = 0; i < _xTrain.Rows; i++)
        {
            for (int j = 0; j < _xTrain.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_xTrain[i, j]));
            }
        }

        // Serialize _yTrain
        writer.Write(_yTrain.Length);
        for (int i = 0; i < _yTrain.Length; i++)
        {
            writer.Write(Convert.ToDouble(_yTrain[i]));
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

        // Deserialize LWR specific data
        _options.Bandwidth = reader.ReadDouble();

        // Deserialize _xTrain
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        _xTrain = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                _xTrain[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Deserialize _yTrain
        int length = reader.ReadInt32();
        _yTrain = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            _yTrain[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}