namespace AiDotNet.Regression;

public class IsotonicRegression<T> : NonLinearRegressionBase<T>
{
    private Vector<T> _xValues;
    private Vector<T> _yValues;

    public IsotonicRegression(NonLinearRegressionOptions? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _xValues = Vector<T>.Empty();
        _yValues = Vector<T>.Empty();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputs(x, y);
        
        // Apply regularization to the input matrix
        var regularizedX = Regularization.RegularizeMatrix(x);
        
        _xValues = regularizedX.GetColumn(0); // Isotonic regression typically works with 1D input
        _yValues = y;
        
        OptimizeModel(regularizedX, _yValues);
    }

    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Implement Pool Adjacent Violators (PAV) algorithm
        var n = y.Length;
        var yhat = new Vector<T>(n, NumOps);
        var w = new Vector<T>(n, NumOps);

        for (int i = 0; i < n; i++)
        {
            yhat[i] = y[i];
            w[i] = NumOps.One;
        }

        bool changed;
        do
        {
            changed = false;
            for (int i = 0; i < n - 1; i++)
            {
                if (NumOps.LessThan(yhat[i + 1], yhat[i]))
                {
                    var weightedMean = NumOps.Divide(
                        NumOps.Add(NumOps.Multiply(w[i], yhat[i]), NumOps.Multiply(w[i + 1], yhat[i + 1])),
                        NumOps.Add(w[i], w[i + 1])
                    );
                    yhat[i] = weightedMean;
                    yhat[i + 1] = weightedMean;
                    w[i] = NumOps.Add(w[i], w[i + 1]);
                    w[i + 1] = w[i];
                    changed = true;
                }
            }
        } while (changed);

        // Apply regularization to the coefficients
        yhat = Regularization.RegularizeCoefficients(yhat);

        SupportVectors = new Matrix<T>(n, 1, NumOps);
        for (int i = 0; i < n; i++)
        {
            SupportVectors[i, 0] = _xValues[i];
        }
        Alphas = yhat;
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows, NumOps);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }

        return predictions;
    }

    protected override T PredictSingle(Vector<T> input)
    {
        var x = input[0]; // Isotonic regression typically works with 1D input
        int index = FindNearestIndex(x);
        return Alphas[index];
    }

    private int FindNearestIndex(T x)
    {
        int left = 0;
        int right = SupportVectors.Rows - 1;

        while (left < right)
        {
            int mid = (left + right) / 2;
            if (NumOps.LessThanOrEquals(SupportVectors[mid, 0], x))
            {
                left = mid + 1;
            }
            else
            {
                right = mid;
            }
        }

        if (left > 0 && NumOps.GreaterThan(NumOps.Abs(NumOps.Subtract(SupportVectors[left - 1, 0], x)),
                                           NumOps.Abs(NumOps.Subtract(SupportVectors[left, 0], x))))
        {
            return left;
        }

        return left - 1;
    }

    protected override ModelType GetModelType()
    {
        return ModelType.IsotonicRegression;
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize IsotonicRegression specific data
        writer.Write(_xValues.Length);
        for (int i = 0; i < _xValues.Length; i++)
        {
            writer.Write(Convert.ToDouble(_xValues[i]));
        }

        writer.Write(_yValues.Length);
        for (int i = 0; i < _yValues.Length; i++)
        {
            writer.Write(Convert.ToDouble(_yValues[i]));
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

        // Deserialize IsotonicRegression specific data
        int xLength = reader.ReadInt32();
        _xValues = new Vector<T>(xLength, NumOps);
        for (int i = 0; i < xLength; i++)
        {
            _xValues[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        int yLength = reader.ReadInt32();
        _yValues = new Vector<T>(yLength, NumOps);
        for (int i = 0; i < yLength; i++)
        {
            _yValues[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}