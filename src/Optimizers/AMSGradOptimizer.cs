namespace AiDotNet.Optimizers;

public class AMSGradOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private AMSGradOptimizerOptions _options;
    private Vector<T>? _m;
    private Vector<T>? _v;
    private Vector<T>? _vHat;
    private int _t;

    public AMSGradOptimizer(
        AMSGradOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new AMSGradOptimizerOptions();
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.LearningRate);
        _t = 0;
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        _m = new Vector<T>(currentSolution.Coefficients.Length);
        _v = new Vector<T>(currentSolution.Coefficients.Length);
        _vHat = new Vector<T>(currentSolution.Coefficients.Length);
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _t++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var newSolution = UpdateSolution(currentSolution, gradient);

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

            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient)
    {
        var newCoefficients = new Vector<T>(currentSolution.Coefficients.Length);
        var beta1 = NumOps.FromDouble(_options.Beta1);
        var beta2 = NumOps.FromDouble(_options.Beta2);
        var oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        var oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);

        for (int i = 0; i < currentSolution.Coefficients.Length; i++)
        {
            // Update biased first moment estimate
            _m![i] = NumOps.Add(NumOps.Multiply(beta1, _m[i]), NumOps.Multiply(oneMinusBeta1, gradient[i]));

            // Update biased second raw moment estimate
            _v![i] = NumOps.Add(NumOps.Multiply(beta2, _v[i]), NumOps.Multiply(oneMinusBeta2, NumOps.Multiply(gradient[i], gradient[i])));

            // Update maximum of second raw moment estimate
            _vHat![i] = MathHelper.Max(_vHat[i], _v[i]);

            // Compute bias-corrected first moment estimate
            var mHat = NumOps.Divide(_m[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));

            // Update parameters
            var update = NumOps.Divide(NumOps.Multiply(CurrentLearningRate, mHat), NumOps.Add(NumOps.Sqrt(_vHat[i]), NumOps.FromDouble(_options.Epsilon)));
            newCoefficients[i] = NumOps.Subtract(currentSolution.Coefficients[i], update);
        }

        return new VectorModel<T>(newCoefficients);
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
        if (options is AMSGradOptimizerOptions amsGradOptions)
        {
            _options = amsGradOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AMSGradOptimizerOptions.");
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

            writer.Write(_t);

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
            _options = JsonConvert.DeserializeObject<AMSGradOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _t = reader.ReadInt32();
        }
    }

    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_AMSGrad_{_options.Beta1}_{_options.Beta2}_{_t}";
    }
}