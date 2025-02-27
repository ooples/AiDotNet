namespace AiDotNet.Optimizers;

public class BFGSOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private BFGSOptimizerOptions _options;
    private Matrix<T>? _inverseHessian;
    private new Vector<T>? _previousGradient;
    private Vector<T>? _previousParameters;
    private int _iteration;

    public BFGSOptimizer(
        BFGSOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new BFGSOptimizerOptions();
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

        _inverseHessian = Matrix<T>.CreateIdentity(currentSolution.Coefficients.Length);
        _previousGradient = null;
        _previousParameters = null;
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var newSolution = UpdateSolution(currentSolution, gradient, inputData);

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

            _previousGradient = gradient;
            _previousParameters = currentSolution.Coefficients;
            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient, OptimizationInputData<T> inputData)
    {
        if (_previousGradient != null && _previousParameters != null)
        {
            UpdateInverseHessian(currentSolution.Coefficients, gradient);
        }

        var direction = _inverseHessian!.Multiply(gradient);
        direction = direction.Transform(x => NumOps.Negate(x));

        var step = LineSearch(currentSolution, direction, gradient, inputData);
        var scaledDirection = direction.Transform(x => NumOps.Multiply(x, step));
        var newCoefficients = currentSolution.Coefficients.Add(scaledDirection);

        return new VectorModel<T>(newCoefficients);
    }

    private void UpdateInverseHessian(Vector<T> currentParameters, Vector<T> currentGradient)
    {
        var s = currentParameters.Subtract(_previousParameters!);
        var y = currentGradient.Subtract(_previousGradient!);

        var rho = NumOps.Divide(NumOps.FromDouble(1), y.DotProduct(s));
        var I = Matrix<T>.CreateIdentity(currentParameters.Length);

        var term1 = I.Subtract(s.OuterProduct(y).Multiply(rho));
        var term2 = I.Subtract(y.OuterProduct(s).Multiply(rho));
        var term3 = s.OuterProduct(s).Multiply(rho);

        _inverseHessian = term1.Multiply(_inverseHessian!).Multiply(term2).Add(term3);
    }

    private T LineSearch(ISymbolicModel<T> currentSolution, Vector<T> direction, Vector<T> gradient, OptimizationInputData<T> inputData)
    {
        var alpha = NumOps.FromDouble(1.0);
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
        if (options is BFGSOptimizerOptions bfgsOptions)
        {
            _options = bfgsOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected BFGSOptimizerOptions.");
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
            _options = JsonConvert.DeserializeObject<BFGSOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
        }
    }

    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_BFGS_{_options.InitialLearningRate}_{_options.Tolerance}_{_iteration}";
    }
}