namespace AiDotNet.TimeSeries;

public class SARIMAModel<T> : TimeSeriesModelBase<T>
{
    private readonly SARIMAOptions<T> _sarimaOptions;
    private Vector<T> _arCoefficients;
    private Vector<T> _maCoefficients;
    private Vector<T> _sarCoefficients;
    private Vector<T> _smaCoefficients;
    private T _constant;
    private readonly int _p, _q, _d, _m, _P, _Q, _D;

    public SARIMAModel(SARIMAOptions<T> options) : base(options)
    {
        _sarimaOptions = options;
        _constant = NumOps.Zero;
        _arCoefficients = Vector<T>.Empty();
        _maCoefficients = Vector<T>.Empty();
        _sarCoefficients = Vector<T>.Empty();
        _smaCoefficients = Vector<T>.Empty();
        _p = _sarimaOptions.P;
        _q = _sarimaOptions.Q;
        _d = _sarimaOptions.D;
        _m = _sarimaOptions.SeasonalPeriod;
        _P = _sarimaOptions.SeasonalP;
        _Q = _sarimaOptions.SeasonalQ;
        _D = _sarimaOptions.SeasonalD;
    }

    public Vector<T> GetARParameters()
    {
        return new Vector<T>(_p);
    }

    public Vector<T> GetMAParameters()
    {
        return new Vector<T>(_q);
    }

    public Vector<T> GetSeasonalARParameters()
    {
        return new Vector<T>(_P);
    }

    public Vector<T> GetSeasonalMAParameters()
    {
        return new Vector<T>(_Q);
    }

    public int GetSeasonalPeriod()
    {
        return _m;
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Step 1: Apply seasonal and non-seasonal differencing
        Vector<T> diffY = ApplyDifferencing(y);

        // Step 2: Estimate non-seasonal AR coefficients
        _arCoefficients = TimeSeriesHelper<T>.EstimateARCoefficients(diffY, _p, MatrixDecompositionType.Qr, NumOps);

        // Step 3: Estimate seasonal AR coefficients
        _sarCoefficients = EstimateSeasonalARCoefficients(diffY);

        // Step 4: Calculate residuals after AR and SAR
        Vector<T> arResiduals = CalculateARSARResiduals(diffY);

        // Step 5: Estimate non-seasonal MA coefficients
        _maCoefficients = TimeSeriesHelper<T>.EstimateMACoefficients(arResiduals, _q, NumOps);

        // Step 6: Estimate seasonal MA coefficients
        _smaCoefficients = EstimateSeasonalMACoefficients(arResiduals);

        // Step 7: Estimate constant term
        _constant = EstimateConstant(diffY);
    }

    private Vector<T> ApplyDifferencing(Vector<T> y)
    {
        Vector<T> result = y;
        
        // Apply seasonal differencing
        for (int i = 0; i < _D; i++)
        {
            result = SeasonalDifference(result, _m);
        }

        // Apply non-seasonal differencing
        result = TimeSeriesHelper<T>.DifferenceSeries(result, _d, NumOps);

        return result;
    }

    private Vector<T> SeasonalDifference(Vector<T> y, int period)
    {
        Vector<T> result = new Vector<T>(y.Length - period, NumOps);
        for (int i = period; i < y.Length; i++)
        {
            result[i - period] = NumOps.Subtract(y[i], y[i - period]);
        }
        return result;
    }

    private Vector<T> EstimateSeasonalARCoefficients(Vector<T> y)
    {
        Matrix<T> X = new Matrix<T>(y.Length - _P * _m, _P, NumOps);
        Vector<T> Y = new Vector<T>(y.Length - _P * _m, NumOps);

        for (int i = _P * _m; i < y.Length; i++)
        {
            for (int j = 0; j < _P; j++)
            {
                X[i - _P * _m, j] = y[i - (j + 1) * _m];
            }
            Y[i - _P * _m] = y[i];
        }

        return MatrixSolutionHelper.SolveLinearSystem(X, Y, MatrixDecompositionType.Qr);
    }

    private Vector<T> CalculateARSARResiduals(Vector<T> y)
    {
        int n = y.Length;
        int maxLag = Math.Max(_p, _P * _m);
        Vector<T> residuals = new Vector<T>(n - maxLag, NumOps);

        for (int i = maxLag; i < n; i++)
        {
            T predicted = NumOps.Zero;
            
            // Non-seasonal AR component
            for (int j = 0; j < _p; j++)
            {
                predicted = NumOps.Add(predicted, NumOps.Multiply(_arCoefficients[j], y[i - j - 1]));
            }

            // Seasonal AR component
            for (int j = 0; j < _P; j++)
            {
                predicted = NumOps.Add(predicted, NumOps.Multiply(_sarCoefficients[j], y[i - (j + 1) * _m]));
            }

            residuals[i - maxLag] = NumOps.Subtract(y[i], predicted);
        }

        return residuals;
    }

    private Vector<T> EstimateSeasonalMACoefficients(Vector<T> residuals)
    {
        Vector<T> smaCoefficients = new Vector<T>(_Q, NumOps);
        for (int i = 0; i < _Q; i++)
        {
            smaCoefficients[i] = TimeSeriesHelper<T>.CalculateAutoCorrelation(residuals, (i + 1) * _m, NumOps);
        }

        return smaCoefficients;
    }

    private T EstimateConstant(Vector<T> y)
    {
        T mean = y.Average();
        T arSum = NumOps.Zero;
        T sarSum = NumOps.Zero;

        for (int i = 0; i < _arCoefficients.Length; i++)
        {
            arSum = NumOps.Add(arSum, _arCoefficients[i]);
        }

        for (int i = 0; i < _sarCoefficients.Length; i++)
        {
            sarSum = NumOps.Add(sarSum, _sarCoefficients[i]);
        }

        return NumOps.Multiply(mean, NumOps.Subtract(NumOps.One, NumOps.Add(arSum, sarSum)));
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> predictions = new(input.Rows, NumOps);
        int maxLag = Math.Max(_p, _P * _m);
        Vector<T> lastObservedValues = new(maxLag, NumOps);
        Vector<T> lastErrors = new(Math.Max(_q, _Q * _m), NumOps);

        for (int i = 0; i < predictions.Length; i++)
        {
            T prediction = _constant;

            // Add non-seasonal AR component
            for (int j = 0; j < _p; j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[j], lastObservedValues[j]));
            }

            // Add seasonal AR component
            for (int j = 0; j < _P; j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_sarCoefficients[j], lastObservedValues[(j + 1) * _m - 1]));
            }

            // Add non-seasonal MA component
            for (int j = 0; j < _q; j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[j], lastErrors[j]));
            }

            // Add seasonal MA component
            for (int j = 0; j < _Q; j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_smaCoefficients[j], lastErrors[(j + 1) * _m - 1]));
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
        // Serialize SARIMA-specific options
        writer.Write(_sarimaOptions.P);
        writer.Write(_sarimaOptions.D);
        writer.Write(_sarimaOptions.Q);
        writer.Write(_sarimaOptions.SeasonalP);
        writer.Write(_sarimaOptions.SeasonalD);
        writer.Write(_sarimaOptions.SeasonalQ);
        writer.Write(_sarimaOptions.MaxIterations);
        writer.Write(Convert.ToDouble(_sarimaOptions.Tolerance));

        // Serialize coefficients
        SerializationHelper<T>.SerializeVector(writer, _arCoefficients);
        SerializationHelper<T>.SerializeVector(writer, _maCoefficients);
        SerializationHelper<T>.SerializeVector(writer, _sarCoefficients);
        SerializationHelper<T>.SerializeVector(writer, _smaCoefficients);
        writer.Write(Convert.ToDouble(_constant));
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Deserialize SARIMA-specific options
        _sarimaOptions.P = reader.ReadInt32();
        _sarimaOptions.D = reader.ReadInt32();
        _sarimaOptions.Q = reader.ReadInt32();
        _sarimaOptions.SeasonalP = reader.ReadInt32();
        _sarimaOptions.SeasonalD = reader.ReadInt32();
        _sarimaOptions.SeasonalQ = reader.ReadInt32();
        _sarimaOptions.MaxIterations = reader.ReadInt32();
        _sarimaOptions.Tolerance = Convert.ToDouble(reader.ReadDouble());

        // Deserialize coefficients
        _arCoefficients = SerializationHelper<T>.DeserializeVector(reader);
        _maCoefficients = SerializationHelper<T>.DeserializeVector(reader);
        _sarCoefficients = SerializationHelper<T>.DeserializeVector(reader);
        _smaCoefficients = SerializationHelper<T>.DeserializeVector(reader);
        _constant = NumOps.FromDouble(reader.ReadDouble());
    }
}