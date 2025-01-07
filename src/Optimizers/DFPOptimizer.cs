namespace AiDotNet.Optimizers;

public class DFPOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private DFPOptimizerOptions _options;
    private Matrix<T> _inverseHessian;
    private new Vector<T> _previousGradient;
    private T _adaptiveLearningRate;

    public DFPOptimizer(
        DFPOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new DFPOptimizerOptions();
        _previousGradient = Vector<T>.Empty();
        _inverseHessian = Matrix<T>.Empty();
        _adaptiveLearningRate = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        _adaptiveLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        _inverseHessian = Matrix<T>.CreateIdentity(currentSolution.Coefficients.Length, NumOps);

        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var direction = CalculateDirection(gradient);
            var newSolution = UpdateSolution(currentSolution, direction, gradient, inputData);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);
            UpdateInverseHessian(currentSolution, newSolution, gradient);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            _previousGradient = gradient;
            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private Vector<T> CalculateDirection(Vector<T> gradient)
    {
        return _inverseHessian.Multiply(gradient).Transform(NumOps.Negate);
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> direction, Vector<T> gradient, OptimizationInputData<T> inputData)
    {
        var stepSize = OptimizerHelper<T>.LineSearch(currentSolution, direction, gradient, inputData, _adaptiveLearningRate);
        var newCoefficients = currentSolution.Coefficients.Add(direction.Multiply(stepSize));

        return currentSolution.UpdateCoefficients(newCoefficients);
    }

    private void UpdateInverseHessian(ISymbolicModel<T> currentSolution, ISymbolicModel<T> newSolution, Vector<T> gradient)
    {
        if (_previousGradient == null)
        {
            _previousGradient = gradient;
            return;
        }

        var s = newSolution.Coefficients.Subtract(currentSolution.Coefficients);
        var y = gradient.Subtract(_previousGradient);

        var sTy = s.DotProduct(y);
        if (NumOps.LessThanOrEquals(sTy, NumOps.Zero))
        {
            return; // Skip update if sTy is not positive
        }

        var term1 = Matrix<T>.OuterProduct(s, s).Divide(sTy);
        var Hy = _inverseHessian.Multiply(y);
        var yTHy = y.DotProduct(Hy);
        var term2 = Matrix<T>.OuterProduct(Hy, Hy).Divide(yTHy);

        _inverseHessian = _inverseHessian.Add(term1).Subtract(term2);
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveLearningRate)
        {
            var improvement = NumOps.Subtract(currentStepData.FitnessScore, previousStepData.FitnessScore);
            var adaptationRate = NumOps.FromDouble(_options.AdaptationRate);

            if (NumOps.GreaterThan(improvement, NumOps.Zero))
            {
                _adaptiveLearningRate = NumOps.Multiply(_adaptiveLearningRate, NumOps.Add(NumOps.One, adaptationRate));
            }
            else
            {
                _adaptiveLearningRate = NumOps.Multiply(_adaptiveLearningRate, NumOps.Subtract(NumOps.One, adaptationRate));
            }

            _adaptiveLearningRate = MathHelper.Clamp(_adaptiveLearningRate, NumOps.FromDouble(_options.MinLearningRate), NumOps.FromDouble(_options.MaxLearningRate));
        }
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is DFPOptimizerOptions dfpOptions)
        {
            _options = dfpOptions;
        }
        else
        {
            throw new ArgumentException("Options must be of type DFPOptimizerOptions", nameof(options));
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

            // Serialize _inverseHessian
            byte[] inverseHessianData = _inverseHessian.Serialize();
            writer.Write(inverseHessianData.Length);
            writer.Write(inverseHessianData);

            // Serialize _previousGradient
            byte[] previousGradientData = _previousGradient.Serialize();
            writer.Write(previousGradientData.Length);
            writer.Write(previousGradientData);

            // Serialize _adaptiveLearningRate
            writer.Write(Convert.ToDouble(_adaptiveLearningRate));

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
            _options = JsonConvert.DeserializeObject<DFPOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize _inverseHessian
            int inverseHessianLength = reader.ReadInt32();
            byte[] inverseHessianData = reader.ReadBytes(inverseHessianLength);
            _inverseHessian = Matrix<T>.Deserialize(inverseHessianData);

            // Deserialize _previousGradient
            int previousGradientLength = reader.ReadInt32();
            byte[] previousGradientData = reader.ReadBytes(previousGradientLength);
            _previousGradient = Vector<T>.Deserialize(previousGradientData);

            // Deserialize _adaptiveLearningRate
            _adaptiveLearningRate = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}