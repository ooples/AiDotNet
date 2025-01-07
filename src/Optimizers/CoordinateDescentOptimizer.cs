namespace AiDotNet.Optimizers;

public class CoordinateDescentOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private CoordinateDescentOptimizerOptions _options;
    private Vector<T> _learningRates;
    private Vector<T> _momentums;
    private Vector<T> _previousUpdate;

    public CoordinateDescentOptimizer(
        CoordinateDescentOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new CoordinateDescentOptimizerOptions();
        _learningRates = Vector<T>.Empty();
        _momentums = Vector<T>.Empty();
        _previousUpdate = Vector<T>.Empty();
    }

    private void InitializeAdaptiveParameters(ISymbolicModel<T> currentSolution)
    {
        base.InitializeAdaptiveParameters();
        int dimensions = currentSolution.Coefficients.Length;
        _learningRates = Vector<T>.CreateDefault(dimensions, NumOps.FromDouble(_options.InitialLearningRate));
        _momentums = Vector<T>.CreateDefault(dimensions, NumOps.FromDouble(_options.InitialMomentum));
        _previousUpdate = Vector<T>.CreateDefault(dimensions, NumOps.Zero);
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        InitializeAdaptiveParameters(currentSolution);

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var newSolution = UpdateSolution(currentSolution, inputData);
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

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, OptimizationInputData<T> inputData)
    {
        var newCoefficients = currentSolution.Coefficients.Copy();

        for (int i = 0; i < newCoefficients.Length; i++)
        {
            var gradient = CalculatePartialDerivative(currentSolution, inputData, i);
            var update = CalculateUpdate(gradient, i);
            newCoefficients[i] = NumOps.Add(newCoefficients[i], update);
        }

        return currentSolution.UpdateCoefficients(newCoefficients);
    }

    private T CalculatePartialDerivative(ISymbolicModel<T> model, OptimizationInputData<T> inputData, int index)
    {
        var epsilon = NumOps.FromDouble(1e-6);
        var originalCoeff = model.Coefficients[index];

        var coefficientsPlus = model.Coefficients.Copy();
        coefficientsPlus[index] = NumOps.Add(originalCoeff, epsilon);
        var modelPlus = model.UpdateCoefficients(coefficientsPlus);

        var coefficientsMinus = model.Coefficients.Copy();
        coefficientsMinus[index] = NumOps.Subtract(originalCoeff, epsilon);
        var modelMinus = model.UpdateCoefficients(coefficientsMinus);

        var lossPlus = CalculateLoss(modelPlus, inputData);
        var lossMinus = CalculateLoss(modelMinus, inputData);

        return NumOps.Divide(NumOps.Subtract(lossPlus, lossMinus), NumOps.Multiply(NumOps.FromDouble(2.0), epsilon));
    }

    private T CalculateUpdate(T gradient, int index)
    {
        var update = NumOps.Add(
            NumOps.Multiply(_learningRates[index], gradient),
            NumOps.Multiply(_momentums[index], _previousUpdate[index])
        );
        _previousUpdate[index] = update;

        return NumOps.Negate(update);
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        var improvement = NumOps.Subtract(currentStepData.FitnessScore, previousStepData.FitnessScore);

        for (int i = 0; i < _learningRates.Length; i++)
        {
            if (NumOps.GreaterThan(improvement, NumOps.Zero))
            {
                _learningRates[i] = NumOps.Multiply(_learningRates[i], NumOps.Add(NumOps.One, NumOps.FromDouble(_options.LearningRateIncreaseRate)));
                _momentums[i] = NumOps.Multiply(_momentums[i], NumOps.Add(NumOps.One, NumOps.FromDouble(_options.MomentumIncreaseRate)));
            }
            else
            {
                _learningRates[i] = NumOps.Multiply(_learningRates[i], NumOps.Subtract(NumOps.One, NumOps.FromDouble(_options.LearningRateDecreaseRate)));
                _momentums[i] = NumOps.Multiply(_momentums[i], NumOps.Subtract(NumOps.One, NumOps.FromDouble(_options.MomentumDecreaseRate)));
            }

            _learningRates[i] = MathHelper.Clamp(_learningRates[i], NumOps.FromDouble(_options.MinLearningRate), NumOps.FromDouble(_options.MaxLearningRate));
            _momentums[i] = MathHelper.Clamp(_momentums[i], NumOps.FromDouble(_options.MinMomentum), NumOps.FromDouble(_options.MaxMomentum));
        }
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is CoordinateDescentOptimizerOptions cdOptions)
        {
            _options = cdOptions;
        }
        else
        {
            throw new ArgumentException("Options must be of type CoordinateDescentOptimizerOptions", nameof(options));
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

            // Serialize _learningRates
            byte[] learningRatesData = _learningRates.Serialize();
            writer.Write(learningRatesData.Length);
            writer.Write(learningRatesData);

            // Serialize _momentums
            byte[] momentumsData = _momentums.Serialize();
            writer.Write(momentumsData.Length);
            writer.Write(momentumsData);

            // Serialize _previousUpdate
            byte[] previousUpdateData = _previousUpdate.Serialize();
            writer.Write(previousUpdateData.Length);
            writer.Write(previousUpdateData);

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
            _options = JsonConvert.DeserializeObject<CoordinateDescentOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize _learningRates
            int learningRatesLength = reader.ReadInt32();
            byte[] learningRatesData = reader.ReadBytes(learningRatesLength);
            _learningRates = Vector<T>.Deserialize(learningRatesData);

            // Deserialize _momentums
            int momentumsLength = reader.ReadInt32();
            byte[] momentumsData = reader.ReadBytes(momentumsLength);
            _momentums = Vector<T>.Deserialize(momentumsData);

            // Deserialize _previousUpdate
            int previousUpdateLength = reader.ReadInt32();
            byte[] previousUpdateData = reader.ReadBytes(previousUpdateLength);
            _previousUpdate = Vector<T>.Deserialize(previousUpdateData);
        }
    }
}