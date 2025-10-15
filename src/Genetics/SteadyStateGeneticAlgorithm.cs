namespace AiDotNet.Genetics;

/// <summary>
/// Implements a steady-state genetic algorithm for evolving machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type produced by the model.</typeparam>
/// <remarks>
/// <para>
/// The SteadyStateGeneticAlgorithm extends the standard genetic algorithm by implementing 
/// an incremental evolution strategy. Unlike traditional genetic algorithms that replace
/// the entire population in each generation, this algorithm only replaces a small portion
/// of the population (the worst-performing individuals) with new offspring in each cycle.
/// This approach maintains more stability in the population and can lead to more gradual,
/// controlled evolution.
/// </para>
/// <para><b>For Beginners:</b> Think of this like a continuous gardening approach rather than seasonal planting.
/// 
/// Imagine a gardening method where:
/// - You have a large garden with many different tomato plants
/// - Instead of replacing all plants each season, you only replace the weakest 10-20%
/// - The healthiest plants remain in your garden across multiple seasons
/// - You focus your breeding efforts on creating replacements for just the weakest plants
/// - This creates a more stable garden that gradually improves over time
/// 
/// This method preserves successful solutions while still allowing innovation,
/// making it well-suited for problems where you want to refine good solutions without
/// risking losing progress through radical changes.
/// </para>
/// </remarks>
public class SteadyStateGeneticAlgorithm<T, TInput, TOutput> :
    StandardGeneticAlgorithm<T, TInput, TOutput>
{
    /// <summary>
    /// The fraction of the population that is replaced in each generation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls how many individuals are replaced in each generation cycle.
    /// It represents the fraction of the population (typically between 0.05 and 0.3) that
    /// will be removed and replaced with new offspring. Lower values result in more gradual
    /// evolution, while higher values make the algorithm behave more like a traditional
    /// generational approach.
    /// </para>
    /// <para><b>For Beginners:</b> This is like deciding what percentage of your garden to replant each season.
    /// 
    /// For example:
    /// - A replacement rate of 0.1 means you replace 10% of your plants each season
    /// - A very low rate (like 0.05 or 5%) makes change happen very slowly but safely
    /// - A higher rate (like 0.3 or 30%) creates more opportunity for improvement but with more disruption
    /// - The ideal rate depends on how quickly you want to see change versus how much stability you need
    /// 
    /// Finding the right replacement rate is crucial for balancing between preserving
    /// good solutions and allowing sufficient exploration of new possibilities.
    /// </para>
    /// </remarks>
    private readonly double _replacementRate;

    /// <summary>
    /// Initializes a new instance of the SteadyStateGeneticAlgorithm class.
    /// </summary>
    /// <param name="modelFactory">Factory function that creates new model instances.</param>
    /// <param name="fitnessCalculator">The calculator used to determine model fitness.</param>
    /// <param name="modelEvaluator">The evaluator used to assess model performance.</param>
    /// <param name="replacementRate">The fraction of the population to replace in each generation (default: 0.1).</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new steady-state genetic algorithm with the specified components.
    /// It extends the standard genetic algorithm but adds the replacementRate parameter, which 
    /// controls how incrementally the population evolves. The default value of 0.1 (10%) is a
    /// common starting point that balances stability and progress.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up your continuous garden breeding program.
    /// 
    /// When starting your program, you specify:
    /// - The same basics needed for any breeding program (as in the standard genetic algorithm)
    /// - Additionally, how aggressive your replacement strategy will be
    /// - A default of 10% means you replace just 1 in 10 plants each cycle
    /// 
    /// This creates a more conservative breeding approach that maintains successful
    /// varieties while slowly introducing improvements.
    /// </para>
    /// </remarks>
    public SteadyStateGeneticAlgorithm(
        Func<IFullModel<T, TInput, TOutput>> modelFactory,
        IFitnessCalculator<T, TInput, TOutput> fitnessCalculator,
        IModelEvaluator<T, TInput, TOutput> modelEvaluator,
        double replacementRate = 0.1)
        : base(modelFactory, fitnessCalculator, modelEvaluator)
    {
        _replacementRate = replacementRate;
    }

    /// <summary>
    /// Creates the next generation by replacing only the worst-performing individuals.
    /// </summary>
    /// <param name="trainingInput">The input data used for training.</param>
    /// <param name="trainingOutput">The expected output data for training.</param>
    /// <param name="validationInput">Optional validation input data.</param>
    /// <param name="validationOutput">Optional validation output data.</param>
    /// <returns>The new population for the next generation.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core steady-state evolution strategy by:
    /// 1. Copying the entire current population
    /// 2. Sorting individuals based on fitness
    /// 3. Removing only the worst-performing individuals (based on the replacementRate)
    /// 4. Creating new offspring through selection, crossover, and mutation to replace them
    /// This approach maintains the best solutions across generations while still allowing
    /// evolutionary progress.
    /// </para>
    /// <para><b>For Beginners:</b> This is like your seasonal garden maintenance process.
    /// 
    /// Each growing cycle:
    /// 1. You evaluate all your existing plants
    /// 2. You identify the weakest, least productive plants
    /// 3. You remove only those underperforming plants
    /// 4. You create new crossbred seedlings to replace just those removed plants
    /// 5. The majority of your garden remains unchanged, especially the best plants
    /// 
    /// This focused approach ensures you don't lose your best varieties while still
    /// creating room for potential improvements through selective breeding.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Gets metadata about the algorithm and its current state.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the algorithm.</returns>
    /// <remarks>
    /// <para>
    /// This method extends the base implementation to include steady-state-specific information
    /// in the metadata. It adds details about the algorithm type, description, and the replacement
    /// rate used for evolution.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a report about your specialized breeding approach.
    /// 
    /// The report includes:
    /// - Basic information about your breeding program (from the base class)
    /// - A description clarifying that you're using a steady-state approach
    /// - The specific replacement rate you're using (what percentage of plants you replace each cycle)
    /// 
    /// This information helps document your breeding program's specific strategy and parameters
    /// for future reference or comparison with other approaches.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetMetaData()
    {
        var metadata = base.GetMetaData();
        metadata.ModelType = ModelType.GeneticAlgorithmRegression;
        metadata.Description = "Model evolved using a steady-state genetic algorithm";
        metadata.AdditionalInfo["ReplacementRate"] = _replacementRate;

        return metadata;
    }
}