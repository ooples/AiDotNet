namespace AiDotNet.Optimizers;

public class SimulatedAnnealingOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random;
    private SimulatedAnnealingOptions _saOptions;
    private T _currentTemperature;

    public SimulatedAnnealingOptimizer(
        SimulatedAnnealingOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        _random = new Random();
        _saOptions = options ?? new SimulatedAnnealingOptions();
        _currentTemperature = NumOps.FromDouble(_saOptions.InitialTemperature);
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        InitializeAdaptiveParameters();
        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        for (int iteration = 0; iteration < _saOptions.MaxIterations; iteration++)
        {
            var newSolution = GenerateNeighborSolution(currentSolution);
            var currentStepData = EvaluateSolution(newSolution, inputData);

            if (AcceptNewSolution(previousStepData.FitnessScore, currentStepData.FitnessScore))
            {
                currentSolution = newSolution;
                UpdateBestSolution(currentStepData, ref bestStepData);
            }

            UpdateAdaptiveParameters(currentStepData, previousStepData);
            _currentTemperature = CoolDown(_currentTemperature);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        UpdateTemperature(currentStepData.FitnessScore, previousStepData.FitnessScore);
        UpdateNeighborGenerationParameters(currentStepData.FitnessScore, previousStepData.FitnessScore);
    }

    private void UpdateTemperature(T currentFitness, T previousFitness)
    {
        if (_fitnessCalculator.IsBetterFitness(currentFitness, previousFitness))
        {
            _currentTemperature = NumOps.Multiply(_currentTemperature, NumOps.FromDouble(_saOptions.CoolingRate));
        }
        else
        {
            _currentTemperature = NumOps.Divide(_currentTemperature, NumOps.FromDouble(_saOptions.CoolingRate));
        }

        _currentTemperature = MathHelper.Clamp(_currentTemperature, 
            NumOps.FromDouble(_saOptions.MinTemperature), 
            NumOps.FromDouble(_saOptions.MaxTemperature));
    }

    private void UpdateNeighborGenerationParameters(T currentFitness, T previousFitness)
    {
        if (_fitnessCalculator.IsBetterFitness(currentFitness, previousFitness))
        {
            _saOptions.NeighborGenerationRange *= 0.95;
        }
        else
        {
            _saOptions.NeighborGenerationRange *= 1.05;
        }

        _saOptions.NeighborGenerationRange = MathHelper.Clamp(_saOptions.NeighborGenerationRange,
            _saOptions.MinNeighborGenerationRange,
            _saOptions.MaxNeighborGenerationRange);
    }

    private T CoolDown(T temperature)
    {
        return NumOps.Multiply(temperature, NumOps.FromDouble(_saOptions.CoolingRate));
    }

    private bool AcceptNewSolution(T currentFitness, T newFitness)
    {
        if (_fitnessCalculator.IsBetterFitness(newFitness, currentFitness))
        {
            return true;
        }

        var acceptanceProbability = Math.Exp(Convert.ToDouble(NumOps.Divide(
            NumOps.Subtract(currentFitness, newFitness),
            _currentTemperature
        )));

        return _random.NextDouble() < acceptanceProbability;
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is SimulatedAnnealingOptions saOptions)
        {
            _saOptions = saOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected SimulatedAnnealingOptions.");
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _saOptions;
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _currentTemperature = NumOps.FromDouble(_saOptions.InitialTemperature);
    }

    private ISymbolicModel<T> GenerateNeighborSolution(ISymbolicModel<T> currentSolution)
    {
        var newCoefficients = new T[currentSolution.Coefficients.Length];
        for (int i = 0; i < newCoefficients.Length; i++)
        {
            var perturbation = NumOps.FromDouble((_random.NextDouble() * 2 - 1) * _saOptions.NeighborGenerationRange);
            newCoefficients[i] = NumOps.Add(currentSolution.Coefficients[i], perturbation);
        }
        return new VectorModel<T>(new Vector<T>(newCoefficients));
    }

    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize SimulatedAnnealingOptions
            string optionsJson = JsonConvert.SerializeObject(_saOptions);
            writer.Write(optionsJson);

            // Serialize current temperature
            writer.Write(Convert.ToDouble(_currentTemperature));

            return ms.ToArray();
        }
    }

    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize SimulatedAnnealingOptions
            string optionsJson = reader.ReadString();
            _saOptions = JsonConvert.DeserializeObject<SimulatedAnnealingOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize current temperature
            _currentTemperature = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}