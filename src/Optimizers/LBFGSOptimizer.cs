namespace AiDotNet.Optimizers;

public class LBFGSOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private LBFGSOptimizerOptions _options;
    private List<Vector<T>> _s;
    private List<Vector<T>> _y;
    private int _iteration;

    public LBFGSOptimizer(
        LBFGSOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new LBFGSOptimizerOptions();
        _s = new List<Vector<T>>();
        _y = new List<Vector<T>>();
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
        _iteration = 0;
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        _s.Clear();
        _y.Clear();
        InitializeAdaptiveParameters();

        Vector<T> previousGradient = Vector<T>.Empty();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var direction = CalculateDirection(gradient);
            var newSolution = UpdateSolution(currentSolution, direction, gradient, inputData);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            UpdateLBFGSMemory(currentSolution.Coefficients, newSolution.Coefficients, gradient, previousGradient);

            previousGradient = gradient;
            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private Vector<T> CalculateDirection(Vector<T> gradient)
    {
        if (_s.Count == 0 || _y.Count == 0)
        {
            return gradient.Transform(x => NumOps.Negate(x));
        }

        var q = new Vector<T>(gradient);
        var alphas = new T[_s.Count];

        for (int i = _s.Count - 1; i >= 0; i--)
        {
            alphas[i] = NumOps.Divide(_s[i].DotProduct(q), _y[i].DotProduct(_s[i]));
            q = q.Subtract(_y[i].Multiply(alphas[i]));
        }

        var gamma = NumOps.Divide(_s[_s.Count - 1].DotProduct(_y[_s.Count - 1]), _y[_s.Count - 1].DotProduct(_y[_s.Count - 1]));
        var z = q.Multiply(gamma);

        for (int i = 0; i < _s.Count; i++)
        {
            var beta = NumOps.Divide(_y[i].DotProduct(z), _y[i].DotProduct(_s[i]));
            z = z.Add(_s[i].Multiply(NumOps.Subtract(alphas[i], beta)));
        }

        return z.Transform(x => NumOps.Negate(x));
    }

    private void UpdateLBFGSMemory(Vector<T> oldSolution, Vector<T> newSolution, Vector<T> gradient, Vector<T> previousGradient)
    {
        var s = newSolution.Subtract(oldSolution);
        var y = gradient.Subtract(previousGradient);

        if (_s.Count >= _options.MemorySize)
        {
            _s.RemoveAt(0);
            _y.RemoveAt(0);
        }

        _s.Add(s);
        _y.Add(y);
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> direction, Vector<T> gradient, OptimizationInputData<T> inputData)
    {
        var step = LineSearch(currentSolution, direction, gradient, inputData);
        var scaledDirection = direction.Transform(x => NumOps.Multiply(x, step));
        var newCoefficients = currentSolution.Coefficients.Add(scaledDirection);

        return new VectorModel<T>(newCoefficients);
    }

    private T LineSearch(ISymbolicModel<T> currentSolution, Vector<T> direction, Vector<T> gradient, OptimizationInputData<T> inputData)
    {
        var alpha = CurrentLearningRate;
        var c1 = NumOps.FromDouble(1e-4);
        var c2 = NumOps.FromDouble(0.9);
        var xTrain = inputData.XTrain;
        var yTrain = inputData.YTrain;

        var initialValue = CalculateLoss(currentSolution, inputData);
        var initialSlope = gradient.DotProduct(direction);

        while (true)
        {
            var newCoefficients = currentSolution.Coefficients.Add(direction.Multiply(alpha));
            var newSolution = new VectorModel<T>(newCoefficients);
            var newValue = CalculateLoss(newSolution, inputData);

            if (NumOps.LessThanOrEquals(newValue, NumOps.Add(initialValue, NumOps.Multiply(NumOps.Multiply(c1, alpha), initialSlope))))
            {
                var newGradient = CalculateGradient(newSolution, xTrain, yTrain);
                var newSlope = newGradient.DotProduct(direction);

                if (NumOps.GreaterThanOrEquals(NumOps.Abs(newSlope), NumOps.Multiply(c2, NumOps.Abs(initialSlope))))
                {
                    return alpha;
                }
            }

            alpha = NumOps.Multiply(alpha, NumOps.FromDouble(0.5));

            if (NumOps.LessThan(alpha, NumOps.FromDouble(1e-10)))
            {
                return NumOps.FromDouble(1e-10);
            }
        }
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveLearningRate)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateIncreaseFactor));
            }
            else
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateDecreaseFactor));
            }

            CurrentLearningRate = MathHelper.Clamp(CurrentLearningRate, 
                NumOps.FromDouble(_options.MinLearningRate), 
                NumOps.FromDouble(_options.MaxLearningRate));
        }
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is LBFGSOptimizerOptions lbfgsOptions)
        {
            _options = lbfgsOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected LBFGSOptimizerOptions.");
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _options;
    }

    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            writer.Write(_iteration);
            writer.Write(_s.Count);
            foreach (var vector in _s)
            {
                writer.Write(vector.Serialize());
            }
            writer.Write(_y.Count);
            foreach (var vector in _y)
            {
                writer.Write(vector.Serialize());
            }

            return ms.ToArray();
        }
    }

    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<LBFGSOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();

            int sCount = reader.ReadInt32();
            _s = new List<Vector<T>>(sCount);
            for (int i = 0; i < sCount; i++)
            {
                int vectorLength = reader.ReadInt32();
                byte[] vectorData = reader.ReadBytes(vectorLength);
                _s.Add(Vector<T>.Deserialize(vectorData));
            }

            int yCount = reader.ReadInt32();
            _y = new List<Vector<T>>(yCount);
            for (int i = 0; i < yCount; i++)
            {
                int vectorLength = reader.ReadInt32();
                byte[] vectorData = reader.ReadBytes(vectorLength);
                _y.Add(Vector<T>.Deserialize(vectorData));
            }
        }
    }
}