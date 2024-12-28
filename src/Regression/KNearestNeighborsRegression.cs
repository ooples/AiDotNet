namespace AiDotNet.Regression;

public class KNearestNeighborsRegression<T> : NonLinearRegressionBase<T>
{
    private readonly KNearestNeighborsOptions _options;
    private Matrix<T> _xTrain;
    private Vector<T> _yTrain;

    public KNearestNeighborsRegression(KNearestNeighborsOptions? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new KNearestNeighborsOptions();
        _xTrain = new Matrix<T>(0, 0, NumOps);
        _yTrain = new Vector<T>(0, NumOps);
    }

    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Apply regularization to the training data
        if (Regularization != null)
        {
            _xTrain = Regularization.RegularizeMatrix(x);
            _yTrain = Regularization.RegularizeCoefficients(y);
        }
        else
        {
            _xTrain = x;
            _yTrain = y;
        }
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        // Apply regularization to the input data if available
        Matrix<T> regularizedInput = Regularization != null ? Regularization.RegularizeMatrix(input) : input;

        var predictions = new Vector<T>(regularizedInput.Rows, NumOps);
        for (int i = 0; i < regularizedInput.Rows; i++)
        {
            predictions[i] = PredictSingle(regularizedInput.GetRow(i));
        }

        return predictions;
    }

    protected override T PredictSingle(Vector<T> input)
    {
        var distances = new List<(int index, T distance)>();

        for (int i = 0; i < _xTrain.Rows; i++)
        {
            T distance = CalculateDistance(input, _xTrain.GetRow(i));
            distances.Add((i, distance));
        }

        var nearestNeighbors = distances
            .OrderBy(x => x.distance)
            .Take(_options.K)
            .ToList();

        T sum = NumOps.Zero;
        foreach (var (index, distance) in nearestNeighbors)
        {
            sum = NumOps.Add(sum, _yTrain[index]);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(_options.K));
    }

    private T CalculateDistance(Vector<T> v1, Vector<T> v2)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            T diff = NumOps.Subtract(v1[i], v2[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sum);
    }

    protected override ModelType GetModelType() => ModelType.KNearestNeighbors;

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize KNN specific data
        writer.Write(_options.K);

        // Serialize training data
        writer.Write(_xTrain.Rows);
        writer.Write(_xTrain.Columns);
        for (int i = 0; i < _xTrain.Rows; i++)
            for (int j = 0; j < _xTrain.Columns; j++)
                writer.Write(Convert.ToDouble(_xTrain[i, j]));

        writer.Write(_yTrain.Length);
        for (int i = 0; i < _yTrain.Length; i++)
            writer.Write(Convert.ToDouble(_yTrain[i]));

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

        // Deserialize KNN specific data
        _options.K = reader.ReadInt32();

        // Deserialize training data
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        _xTrain = new Matrix<T>(rows, cols, NumOps);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                _xTrain[i, j] = NumOps.FromDouble(reader.ReadDouble());

        int yLength = reader.ReadInt32();
        _yTrain = new Vector<T>(yLength, NumOps);
        for (int i = 0; i < yLength; i++)
            _yTrain[i] = NumOps.FromDouble(reader.ReadDouble());

        // Apply regularization to the deserialized data if available
        if (Regularization != null)
        {
            _xTrain = Regularization.RegularizeMatrix(_xTrain);
            _yTrain = Regularization.RegularizeCoefficients(_yTrain);
        }
    }
}