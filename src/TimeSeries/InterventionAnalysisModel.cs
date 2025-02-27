namespace AiDotNet.TimeSeries;

public class InterventionAnalysisModel<T> : TimeSeriesModelBase<T>
{
    private readonly InterventionAnalysisOptions<T> _iaOptions;
    private Vector<T> _arParameters;
    private Vector<T> _maParameters;
    private List<InterventionEffect<T>> _interventionEffects;
    private Vector<T> _residuals;
    private Vector<T> _fitted;
    private readonly IOptimizer<T> _optimizer;
    private Vector<T> _y;

    public InterventionAnalysisModel(InterventionAnalysisOptions<T>? options = null) : base(options ?? new())
    {
        _iaOptions = options ?? new InterventionAnalysisOptions<T>();
        _optimizer = _iaOptions.Optimizer ?? new LBFGSOptimizer<T>();
        _interventionEffects = [];
        _arParameters = Vector<T>.Empty();
        _maParameters = Vector<T>.Empty();
        _residuals = Vector<T>.Empty();
        _fitted = Vector<T>.Empty();
        _y = Vector<T>.Empty();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Input matrix rows must match output vector length.");
        }

        _y = y;

        InitializeParameters();
        OptimizeParameters(x, _y);
        ComputeResiduals(x, _y);
    }

    private void InitializeParameters()
    {
        int p = _iaOptions.AROrder;
        int q = _iaOptions.MAOrder;

        _arParameters = new Vector<T>(p);
        _maParameters = new Vector<T>(q);

        // Initialize with small random values
        Random rand = new();
        for (int i = 0; i < p; i++) _arParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
        for (int i = 0; i < q; i++) _maParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);

        // Initialize intervention effects
        foreach (var intervention in _iaOptions.Interventions)
        {
            _interventionEffects.Add(new InterventionEffect<T>
            {
                StartIndex = intervention.StartIndex,
                Duration = intervention.Duration,
                Effect = rand.NextDouble() * 0.1
            });
        }
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

        // Update intervention effects
        for (int i = 0; i < _interventionEffects.Count; i++)
        {
            _interventionEffects[i].Effect = Convert.ToDouble(optimizedParameters[paramIndex++]);
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
            predictions[i] = PredictSingle(predictions, i);
        }

        return predictions;
    }

    private T PredictSingle(Vector<T> predictions, int index)
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

        // Add intervention effects
        foreach (var effect in _interventionEffects)
        {
            if (index >= effect.StartIndex && (effect.Duration == 0 || index < effect.StartIndex + effect.Duration))
            {
                prediction = NumOps.Add(prediction, NumOps.FromDouble(effect.Effect));
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

    public Dictionary<string, double> GetInterventionEffects()
    {
        return _interventionEffects.ToDictionary(
            effect => $"Intervention_{effect.StartIndex}_{effect.Duration}",
            effect => effect.Effect
        );
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write model parameters
        SerializationHelper<T>.SerializeVector(writer, _arParameters);
        SerializationHelper<T>.SerializeVector(writer, _maParameters);
            
        // Write intervention effects
        writer.Write(_interventionEffects.Count);
        foreach (var effect in _interventionEffects)
        {
            writer.Write(effect.StartIndex);
            writer.Write(effect.Duration);
            writer.Write(Convert.ToDouble(effect.Effect));
        }

        // Write options
        writer.Write(_iaOptions.AROrder);
        writer.Write(_iaOptions.MAOrder);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read model parameters
        _arParameters = SerializationHelper<T>.DeserializeVector(reader);
        _maParameters = SerializationHelper<T>.DeserializeVector(reader);

        // Read intervention effects
        int effectCount = reader.ReadInt32();
        _interventionEffects = new List<InterventionEffect<T>>();
        for (int i = 0; i < effectCount; i++)
        {
            _interventionEffects.Add(new InterventionEffect<T>
            {
                StartIndex = reader.ReadInt32(),
                Duration = reader.ReadInt32(),
                Effect = reader.ReadDouble()
            });
        }

        // Read options
        _iaOptions.AROrder = reader.ReadInt32();
        _iaOptions.MAOrder = reader.ReadInt32();
    }
}