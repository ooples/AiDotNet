namespace AiDotNet.Genetics;

public class SteadyStateGeneticAlgorithm<T, TInput, TOutput> :
    StandardGeneticAlgorithm<T, TInput, TOutput>
{
    private readonly double _replacementRate;

    public SteadyStateGeneticAlgorithm(
        Func<IFullModel<T, TInput, TOutput>> modelFactory,
        IFitnessCalculator<T, TInput, TOutput> fitnessCalculator,
        double replacementRate = 0.1)
        : base(modelFactory, fitnessCalculator)
    {
        _replacementRate = replacementRate;
    }

    protected override ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> CreateNextGeneration(
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default)
    {
        // Clone the current population
        var newPopulation = Population
            .Select(p => p.Clone() as ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>)
            .Where(p => p != null)
            .ToList();

        // Sort by fitness
        var sortedPopulation = newPopulation
            .OrderByDescending(i => FitnessCalculator.IsHigherScoreBetter ?
                i!.GetFitness() : InvertFitness(i!.GetFitness()))
            .ToList();

        // Determine how many individuals to replace
        int replacementCount = Math.Max(1, (int)(GeneticParams.PopulationSize * _replacementRate));

        // Remove the worst individuals
        int removeCount = Math.Min(replacementCount, sortedPopulation.Count);
        sortedPopulation.RemoveRange(sortedPopulation.Count - removeCount, removeCount);

        // Create new offspring to replace them
        while (sortedPopulation.Count < GeneticParams.PopulationSize)
        {
            // Select parents
            var parents = Select(2, GeneticParams.SelectionMethod).ToList();
            if (parents.Count < 2)
            {
                continue;
            }

            // Crossover
            var offspring = Crossover(parents[0], parents[1], GeneticParams.CrossoverRate);

            // Mutation and evaluation
            foreach (var child in offspring)
            {
                var mutated = Mutate(child, GeneticParams.MutationRate);
                EvaluateIndividual(mutated, trainingInput, trainingOutput, validationInput, validationOutput);
                sortedPopulation.Add(mutated);

                if (sortedPopulation.Count >= GeneticParams.PopulationSize)
                {
                    break;
                }
            }
        }

        return sortedPopulation!;
    }

    public override ModelMetadata<T> GetMetaData()
    {
        var metadata = base.GetMetaData();
        metadata.ModelType = ModelType.GeneticAlgorithmRegression;
        metadata.Description = "Model evolved using a steady-state genetic algorithm";
        metadata.AdditionalInfo["ReplacementRate"] = _replacementRate.ToString();

        return metadata;
    }
}
