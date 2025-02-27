namespace AiDotNet.Optimizers;

public class RootMeanSquarePropagationOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private Vector<T> _squaredGradient;
    private int _t;
    private RootMeanSquarePropagationOptimizerOptions _options;

    public RootMeanSquarePropagationOptimizer(
        RootMeanSquarePropagationOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        _t = 0;
        _squaredGradient = Vector<T>.Empty();
        _options = options ?? new();
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        _squaredGradient = new Vector<T>(currentSolution.Coefficients.Length);
        _t = 0;
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _t++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            gradient = ApplyMomentum(gradient);
            var newSolution = UpdateSolution(currentSolution, gradient);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                break;
            }

            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    public override Vector<T> UpdateVector(Vector<T> parameters, Vector<T> gradient)
    {
        for (int i = 0; i < parameters.Length; i++)
        {
            var squaredGrad = NumOps.Multiply(gradient[i], gradient[i]);
            _squaredGradient[i] = NumOps.Add(NumOps.Multiply(NumOps.FromDouble(_options.Decay), _squaredGradient[i]), NumOps.Multiply(NumOps.FromDouble(1 - _options.Decay), squaredGrad));
            
            var adaptiveLearningRate = CurrentLearningRate;
            var denominator = NumOps.Add(NumOps.Sqrt(_squaredGradient[i]), NumOps.FromDouble(_options.Epsilon));
            var update = NumOps.Divide(NumOps.Multiply(adaptiveLearningRate, gradient[i]), denominator);
            
            parameters[i] = NumOps.Subtract(parameters[i], update);
        }

        return parameters;
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient)
    {
        for (int i = 0; i < gradient.Length; i++)
        {
            var gradientSquared = NumOps.Multiply(gradient[i], gradient[i]);
            _squaredGradient[i] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(_options.Decay), _squaredGradient[i]),
                NumOps.Multiply(NumOps.FromDouble(1 - _options.Decay), gradientSquared)
            );

            var update = NumOps.Divide(
                NumOps.Multiply(CurrentLearningRate, gradient[i]),
                NumOps.Add(NumOps.Sqrt(_squaredGradient[i]), NumOps.FromDouble(_options.Epsilon))
            );

            currentSolution.Coefficients[i] = NumOps.Subtract(currentSolution.Coefficients[i], update);
        }

        return currentSolution;
    }

    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_RMSprop_{CurrentLearningRate}_{_options.Decay}_{_options.Epsilon}_{_t}";
    }

    public override void Reset()
    {
        base.Reset();
        _t = 0;
        _squaredGradient = Vector<T>.Empty();
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _options;
    }

    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize RootMeanSquarePropagationOptimizer specific data
        writer.Write(_t);
        writer.Write(JsonConvert.SerializeObject(_squaredGradient));
        writer.Write(JsonConvert.SerializeObject(_options));

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize RootMeanSquarePropagationOptimizer specific data
        _t = reader.ReadInt32();
        _squaredGradient = JsonConvert.DeserializeObject<Vector<T>>(reader.ReadString())
            ?? throw new InvalidOperationException("Failed to deserialize _squaredGradient.");
        _options = JsonConvert.DeserializeObject<RootMeanSquarePropagationOptimizerOptions>(reader.ReadString())
            ?? throw new InvalidOperationException("Failed to deserialize _options.");
    }
}