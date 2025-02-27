namespace AiDotNet.Optimizers;

public class FTRLOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private FTRLOptimizerOptions _options;
    private Vector<T>? _z;
    private Vector<T>? _n;
    private int _t;

    public FTRLOptimizer(
        FTRLOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new FTRLOptimizerOptions();
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.Alpha);
        _t = 0;
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        _z = new Vector<T>(currentSolution.Coefficients.Length);
        _n = new Vector<T>(currentSolution.Coefficients.Length);
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
        var alpha = NumOps.FromDouble(_options.Alpha);
        var beta = NumOps.FromDouble(_options.Beta);
        var lambda1 = NumOps.FromDouble(_options.Lambda1);
        var lambda2 = NumOps.FromDouble(_options.Lambda2);

        for (int i = 0; i < currentSolution.Coefficients.Length; i++)
        {
            var sigma = NumOps.Divide(
                NumOps.Subtract(NumOps.Sqrt(NumOps.Add(_n![i], NumOps.Multiply(gradient[i], gradient[i]))), NumOps.Sqrt(_n[i])),
                alpha
            );
            _z![i] = NumOps.Add(_z[i], NumOps.Subtract(gradient[i], NumOps.Multiply(sigma, currentSolution.Coefficients[i])));
            _n![i] = NumOps.Add(_n[i], NumOps.Multiply(gradient[i], gradient[i]));

            var sign = NumOps.SignOrZero(_z[i]);
            if (NumOps.GreaterThan(NumOps.Abs(_z[i]), lambda1))
            {
                newCoefficients[i] = NumOps.Divide(
                    NumOps.Multiply(
                        NumOps.Subtract(lambda1, _z[i]),
                        sign
                    ),
                    NumOps.Add(
                        NumOps.Multiply(lambda2, NumOps.FromDouble(1 + _options.Beta)),
                        NumOps.Divide(
                            NumOps.Sqrt(_n[i]),
                            alpha
                        )
                    )
                );
            }
            else
            {
                newCoefficients[i] = NumOps.FromDouble(0);
            }
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
        if (options is FTRLOptimizerOptions ftrlOptions)
        {
            _options = ftrlOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected FTRLOptimizerOptions.");
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
            _options = JsonConvert.DeserializeObject<FTRLOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _t = reader.ReadInt32();
        }
    }

    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_FTRL_{_options.Alpha}_{_options.Beta}_{_options.Lambda1}_{_options.Lambda2}_{_t}";
    }
}