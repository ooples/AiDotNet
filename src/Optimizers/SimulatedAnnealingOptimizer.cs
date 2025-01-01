namespace AiDotNet.Optimizers;

public class SimulatedAnnealingOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random;
    private SimulatedAnnealingOptions _saOptions;

    public SimulatedAnnealingOptimizer(
        SimulatedAnnealingOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator)
    {
        _random = new Random();
        _saOptions = options ?? new SimulatedAnnealingOptions();
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        int dimensions = inputData.XTrain.Columns;
        var currentSolution = InitializeRandomSolution(dimensions);
        var bestStepData = new OptimizationStepData<T>();
        T temperature = NumOps.FromDouble(_saOptions.InitialTemperature);

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            var newSolution = PerturbSolution(currentSolution);
            var currentStepData = PrepareAndEvaluateSolution(newSolution, inputData);

            if (AcceptSolution(currentStepData.FitnessScore, bestStepData.FitnessScore, temperature))
            {
                currentSolution = newSolution;
                UpdateBestSolution(currentStepData, ref bestStepData);
            }

            temperature = NumOps.Multiply(temperature, NumOps.FromDouble(_saOptions.CoolingRate));

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break; // Early stopping criteria met, exit the loop
            }
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private ISymbolicModel<T> PerturbSolution(ISymbolicModel<T> solution)
    {
        if (solution is ExpressionTree<T> expressionTree)
        {
            return expressionTree.Mutate(_saOptions.MutationRate, NumOps);
        }
        else if (solution is VectorModel<T> vectorModel)
        {
            var newSolution = vectorModel.Coefficients.Copy();
            for (int i = 0; i < newSolution.Length; i++)
            {
                if (_random.NextDouble() < _saOptions.MutationRate)
                {
                    newSolution[i] = NumOps.FromDouble(_random.NextDouble() * 2 - 1); // Random value between -1 and 1
                }
            }
            return new VectorModel<T>(newSolution);
        }
        else
        {
            throw new ArgumentException("Unsupported model type");
        }
    }

    private bool AcceptSolution(T newFitness, T currentFitness, T temperature)
    {
        if (NumOps.GreaterThan(newFitness, currentFitness))
        {
            return true;
        }

        T probability = NumOps.Exp(NumOps.Divide(NumOps.Subtract(newFitness, currentFitness), temperature));
        return NumOps.GreaterThan(NumOps.FromDouble(_random.NextDouble()), probability);
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
        }
    }
}