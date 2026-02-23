namespace AiDotNet.Genetics;

public class AdaptiveGeneticAlgorithm<T, TInput, TOutput> :
    StandardGeneticAlgorithm<T, TInput, TOutput>
{
    private readonly double _minMutationRate;
    private readonly double _maxMutationRate;
    private readonly double _minCrossoverRate;
    private readonly double _maxCrossoverRate;
    private double _currentMutationRate;
    private double _currentCrossoverRate;

    public AdaptiveGeneticAlgorithm(
        Func<IFullModel<T, TInput, TOutput>> modelFactory,
        IFitnessCalculator<T, TInput, TOutput> fitnessCalculator,
        double minMutationRate = 0.001,
        double maxMutationRate = 0.5,
        double minCrossoverRate = 0.4,
        double maxCrossoverRate = 0.95)
        : base(modelFactory, fitnessCalculator)
    {
        _minMutationRate = minMutationRate;
        _maxMutationRate = maxMutationRate;
        _minCrossoverRate = minCrossoverRate;
        _maxCrossoverRate = maxCrossoverRate;

        // Initialize with default rates
        _currentMutationRate = (minMutationRate + maxMutationRate) / 2;
        _currentCrossoverRate = (minCrossoverRate + maxCrossoverRate) / 2;
    }

    public override EvolutionStats<T, TInput, TOutput> Evolve(
        int generations,
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default,
        Func<EvolutionStats<T, TInput, TOutput>, bool>? stopCriteria = null)
    {
        // Initialize rates from genetic parameters
        _currentMutationRate = GeneticParams.MutationRate;
        _currentCrossoverRate = GeneticParams.CrossoverRate;

        // Use base Evolve method
        return base.Evolve(generations, trainingInput, trainingOutput, validationInput, validationOutput, stopCriteria);
    }

    protected override void UpdateEvolutionStats()
    {
        base.UpdateEvolutionStats();

        // Update the parameters based on the diversity and improvement rate
        AdaptParameters();
    }

    private void AdaptParameters()
    {
        // Get current diversity and improvement status
        double diversity = Convert.ToDouble(CurrentStats.Diversity);
        bool improved = CurrentStats.ImprovedInLastGeneration;
        int stagnantGenerations = CurrentStats.GenerationsSinceImprovement;

        // Calculate normalized diversity (assuming reasonable range)
        double normalizedDiversity = Math.Min(1.0, Math.Max(0.0, diversity / 100.0));

        // Adapt mutation rate based on diversity
        if (normalizedDiversity < 0.2)
        {
            // Low diversity - increase mutation rate
            _currentMutationRate = Math.Min(_maxMutationRate, _currentMutationRate * 1.2);
        }
        else if (normalizedDiversity > 0.8)
        {
            // High diversity - decrease mutation rate
            _currentMutationRate = Math.Max(_minMutationRate, _currentMutationRate * 0.8);
        }

        // Adapt crossover rate based on improvement
        if (!improved)
        {
            if (stagnantGenerations > 5)
            {
                // Stagnation - increase crossover rate
                _currentCrossoverRate = Math.Min(_maxCrossoverRate, _currentCrossoverRate * 1.1);
            }
        }
        else
        {
            // Improvement - keep or slightly decrease crossover rate
            _currentCrossoverRate = Math.Max(_minCrossoverRate, _currentCrossoverRate * 0.99);
        }

        // Update GeneticParams with new rates
        GeneticParams.MutationRate = _currentMutationRate;
        GeneticParams.CrossoverRate = _currentCrossoverRate;
    }

    public override ModelMetadata<T> GetMetaData()
    {
        var metadata = base.GetMetaData();
        metadata.ModelType = ModelType.GeneticAlgorithmRegression;
        metadata.Description = "Model evolved using an adaptive genetic algorithm";
        metadata.AdditionalInfo["MinMutationRate"] = _minMutationRate.ToString();
        metadata.AdditionalInfo["MaxMutationRate"] = _maxMutationRate.ToString();
        metadata.AdditionalInfo["MinCrossoverRate"] = _minCrossoverRate.ToString();
        metadata.AdditionalInfo["MaxCrossoverRate"] = _maxCrossoverRate.ToString();
        metadata.AdditionalInfo["FinalMutationRate"] = _currentMutationRate.ToString();
        metadata.AdditionalInfo["FinalCrossoverRate"] = _currentCrossoverRate.ToString();

        return metadata;
    }
}
