namespace AiDotNet.TimeSeries;

public class TransferFunctionModel<T> : TimeSeriesModelBase<T>
{
    private readonly TransferFunctionOptions<T> _tfOptions;
    private Vector<T> _arParameters;
    private Vector<T> _maParameters;
    private Vector<T> _inputLags;
    private Vector<T> _outputLags;
    private Vector<T> _residuals;
    private Vector<T> _fitted;
    private Vector<T> _y;
    private readonly IOptimizer<T> _optimizer;

    public TransferFunctionModel(TransferFunctionOptions<T>? options = null) : base(options ?? new())
    {
        _tfOptions = options ?? new TransferFunctionOptions<T>();
        _optimizer = _tfOptions.Optimizer ?? new LBFGSOptimizer<T>();
        _y = Vector<T>.Empty();
        _arParameters = Vector<T>.Empty();
        _maParameters = Vector<T>.Empty();
        _inputLags = Vector<T>.Empty();
        _outputLags = Vector<T>.Empty();
        _residuals = Vector<T>.Empty();
        _fitted = Vector<T>.Empty();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Input matrix rows must match output vector length.");
        }

        int n = y.Length;
        _y = y;

        InitializeParameters();
        OptimizeParameters(x, y);
        ComputeResiduals(x, y);
    }

    private void InitializeParameters()
    {
        int p = _tfOptions.AROrder;
        int q = _tfOptions.MAOrder;
        int r = _tfOptions.InputLagOrder;
        int s = _tfOptions.OutputLagOrder;

        _arParameters = new Vector<T>(p);
        _maParameters = new Vector<T>(q);
        _inputLags = new Vector<T>(r);
        _outputLags = new Vector<T>(s);

        // Initialize with small random values
        Random rand = new Random();
        for (int i = 0; i < p; i++) _arParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
        for (int i = 0; i < q; i++) _maParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
        for (int i = 0; i < r; i++) _inputLags[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
        for (int i = 0; i < s; i++) _outputLags[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
    }

    private void OptimizeParameters(Matrix<T> x, Vector<T> y)
    {
        var inputData = new OptimizationInputData<T>
        {
            XTrain = x,
            YTrain = y
        };

        OptimizationResult<T> result = _optimizer.Optimize(inputData);
        UpdateModelParameters(result.BestSolution.Coefficients);
    }

    private void UpdateModelParameters(Vector<T> optimizedParameters)
    {
        int paramIndex = 0;

        // Update AR parameters
        for (int i = 0; i < _arParameters.Length; i++)
        {
            _arParameters[i] = optimizedParameters[paramIndex++];
        }

        // Update MA parameters
        for (int i = 0; i < _maParameters.Length; i++)
        {
            _maParameters[i] = optimizedParameters[paramIndex++];
        }

        // Update input lag parameters
        for (int i = 0; i < _inputLags.Length; i++)
        {
            _inputLags[i] = optimizedParameters[paramIndex++];
        }

        // Update output lag parameters
        for (int i = 0; i < _outputLags.Length; i++)
        {
            _outputLags[i] = optimizedParameters[paramIndex++];
        }
    }

    private void ComputeResiduals(Matrix<T> x, Vector<T> y)
    {
        _fitted = Predict(x);
        _residuals = new Vector<T>(y.Length);

        for (int i = 0; i < y.Length; i++)
        {
            _residuals[i] = NumOps.Subtract(y[i], _fitted[i]);
        }
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        int n = input.Rows;
        Vector<T> predictions = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            predictions[i] = PredictSingle(input, predictions, i);
        }

        return predictions;
    }

    private T PredictSingle(Matrix<T> x, Vector<T> predictions, int index)
    {
        T prediction = NumOps.Zero;

        // Add AR terms
        for (int i = 0; i < _arParameters.Length; i++)
        {
            if (index - i - 1 >= 0)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_arParameters[i], predictions[index - i - 1]));
            }
        }

        // Add MA terms
        for (int i = 0; i < _maParameters.Length; i++)
        {
            if (index - i - 1 >= 0 && _residuals != null)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_maParameters[i], _residuals[index - i - 1]));
            }
        }

        // Add input lag terms
        for (int i = 0; i < _inputLags.Length; i++)
        {
            if (index - i >= 0)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_inputLags[i], x[index - i, 0]));
            }
        }

        // Add output lag terms
        for (int i = 0; i < _outputLags.Length; i++)
        {
            if (index - i - 1 >= 0)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_outputLags[i], _y[index - i - 1]));
            }
        }

        return prediction;
    }

    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = new Dictionary<string, T>();

        // Mean Absolute Error (MAE)
        metrics["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions);

        // Mean Squared Error (MSE)
        metrics["MSE"] = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions);

        // Root Mean Squared Error (RMSE)
        metrics["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions);

        // R-squared (R2)
        metrics["R2"] = StatisticsHelper<T>.CalculateR2(yTest, predictions);

        return metrics;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write model parameters
        SerializationHelper<T>.SerializeVector(writer, _arParameters);
        SerializationHelper<T>.SerializeVector(writer, _maParameters);
        SerializationHelper<T>.SerializeVector(writer, _inputLags);
        SerializationHelper<T>.SerializeVector(writer, _outputLags);

        // Write options
        writer.Write(_tfOptions.AROrder);
        writer.Write(_tfOptions.MAOrder);
        writer.Write(_tfOptions.InputLagOrder);
        writer.Write(_tfOptions.OutputLagOrder);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read model parameters
        _arParameters = SerializationHelper<T>.DeserializeVector(reader);
        _maParameters = SerializationHelper<T>.DeserializeVector(reader);
        _inputLags = SerializationHelper<T>.DeserializeVector(reader);
        _outputLags = SerializationHelper<T>.DeserializeVector(reader);

        // Read options
        _tfOptions.AROrder = reader.ReadInt32();
        _tfOptions.MAOrder = reader.ReadInt32();
        _tfOptions.InputLagOrder = reader.ReadInt32();
        _tfOptions.OutputLagOrder = reader.ReadInt32();
    }
}