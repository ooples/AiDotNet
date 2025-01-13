namespace AiDotNet.TimeSeries;

public class ARIMAModel<T> : TimeSeriesModelBase<T>
{
    private ARIMAOptions<T> _arimaOptions;
    private Vector<T> _arCoefficients;
    private Vector<T> _maCoefficients;
    private T _constant;

    public ARIMAModel(ARIMAOptions<T>? options = null) : base(options ?? new())
    {
        _arimaOptions = options ?? new();
        _constant = NumOps.Zero;
        _arCoefficients = Vector<T>.Empty();
        _maCoefficients = Vector<T>.Empty();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int p = _arimaOptions.P; // AR order
        int d = _arimaOptions.D;
        int q = _arimaOptions.Q;

        // Step 1: Difference the series
        Vector<T> diffY = TimeSeriesHelper<T>.DifferenceSeries(y, d, NumOps);

        // Step 2: Estimate AR coefficients
        _arCoefficients = TimeSeriesHelper<T>.EstimateARCoefficients(diffY, p, MatrixDecompositionType.Qr, NumOps);

        // Step 3: Estimate MA coefficients
        Vector<T> arResiduals = TimeSeriesHelper<T>.CalculateARResiduals(diffY, _arCoefficients, NumOps);
        _maCoefficients = TimeSeriesHelper<T>.EstimateMACoefficients(arResiduals, q, NumOps);

        // Step 4: Estimate constant term
        _constant = EstimateConstant(diffY, _arCoefficients, _maCoefficients);
    }

    private T EstimateConstant(Vector<T> y, Vector<T> arCoefficients, Vector<T> maCoefficients)
    {
        T mean = y.Average();
        T arSum = NumOps.Zero;
        for (int i = 0; i < arCoefficients.Length; i++)
        {
            arSum = NumOps.Add(arSum, arCoefficients[i]);
        }

        return NumOps.Multiply(mean, NumOps.Subtract(NumOps.One, arSum));
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> predictions = new(input.Rows, NumOps);
        Vector<T> lastObservedValues = new(_options.LagOrder, NumOps);
        Vector<T> lastErrors = new(_maCoefficients.Length, NumOps);

        for (int i = 0; i < predictions.Length; i++)
        {
            T prediction = _constant;

            // Add AR component
            for (int j = 0; j < _arCoefficients.Length; j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[j], lastObservedValues[j]));
            }

            // Add MA component
            for (int j = 0; j < _maCoefficients.Length; j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[j], lastErrors[j]));
            }

            predictions[i] = prediction;

            // Update last observed values and errors for next prediction
            for (int j = lastObservedValues.Length - 1; j > 0; j--)
            {
                lastObservedValues[j] = lastObservedValues[j - 1];
            }
            lastObservedValues[0] = prediction;

            for (int j = lastErrors.Length - 1; j > 0; j--)
            {
                lastErrors[j] = lastErrors[j - 1];
            }
            lastErrors[0] = NumOps.Zero; // Assume zero error for future predictions
        }

        return predictions;
    }

    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = [];

        // Calculate MSE
        T mse = StatisticsHelper<T>.CalculateMeanSquaredError(predictions, yTest);
        metrics["MSE"] = mse;

        // Calculate RMSE
        T rmse = NumOps.Sqrt(mse);
        metrics["RMSE"] = rmse;

        // Calculate MAE
        T mae = StatisticsHelper<T>.CalculateMeanAbsoluteError(predictions, yTest);
        metrics["MAE"] = mae;

        return metrics;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write ARIMA-specific options
        writer.Write(_arimaOptions.P);
        writer.Write(_arimaOptions.D);
        writer.Write(_arimaOptions.Q);

        // Write constant
        writer.Write(Convert.ToDouble(_constant));

        // Write AR coefficients
        writer.Write(_arCoefficients.Length);
        for (int i = 0; i < _arCoefficients.Length; i++)
        {
            writer.Write(Convert.ToDouble(_arCoefficients[i]));
        }

        // Write MA coefficients
        writer.Write(_maCoefficients.Length);
        for (int i = 0; i < _maCoefficients.Length; i++)
        {
            writer.Write(Convert.ToDouble(_maCoefficients[i]));
        }
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read ARIMA-specific options
        int p = reader.ReadInt32();
        int d = reader.ReadInt32();
        int q = reader.ReadInt32();
        _arimaOptions = new ARIMAOptions<T>
        {
            P = p,
            D = d,
            Q = q
        };

        // Read constant
        _constant = NumOps.FromDouble(reader.ReadDouble());

        // Read AR coefficients
        int arLength = reader.ReadInt32();
        _arCoefficients = new Vector<T>(arLength, NumOps);
        for (int i = 0; i < arLength; i++)
        {
            _arCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Read MA coefficients
        int maLength = reader.ReadInt32();
        _maCoefficients = new Vector<T>(maLength, NumOps);
        for (int i = 0; i < maLength; i++)
        {
            _maCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}