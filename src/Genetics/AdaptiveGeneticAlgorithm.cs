namespace AiDotNet.Genetics;

/// <summary>
/// Implements an adaptive genetic algorithm that automatically adjusts mutation and crossover rates based on population diversity and fitness improvement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
/// <remarks>
/// <para>
/// This class extends the standard genetic algorithm by dynamically adjusting genetic parameters during the evolution process.
/// It monitors population diversity and improvement rates to adapt mutation and crossover rates, helping to balance exploration and exploitation.
/// </para>
/// <para><b>For Beginners:</b> Imagine a gardener who changes their approach based on how well their plants are growing.
/// 
/// A standard genetic algorithm uses fixed settings throughout the entire process, like a gardener who waters plants 
/// the same amount every day regardless of conditions. This adaptive algorithm, however, is like a smart gardener who:
/// 
/// - Adds more fertilizer (increases mutation) when all plants look too similar (low diversity)
/// - Waters less frequently (decreases mutation) when there's already a good variety of plants
/// - Changes planting patterns (adjusts crossover) when plants haven't improved for several generations
/// 
/// This adaptive approach helps the algorithm avoid getting stuck in suboptimal solutions and often 
/// finds better results with less manual parameter tuning.
/// </para>
/// </remarks>
public class AdaptiveGeneticAlgorithm<T, TInput, TOutput> :
    StandardGeneticAlgorithm<T, TInput, TOutput>
{
    /// <summary>
    /// The minimum allowed mutation rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value sets a lower bound on how small the mutation rate can become during adaptation.
    /// A minimum rate ensures that some level of exploration always occurs, even when the algorithm appears to be converging.
    /// </para>
    /// <para><b>For Beginners:</b> This is like ensuring the gardener always adds at least a tiny bit of fertilizer, 
    /// even when plants seem to be growing well, to maintain some genetic diversity.
    /// </para>
    /// </remarks>
    private readonly double _minMutationRate;

    /// <summary>
    /// The maximum allowed mutation rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value sets an upper bound on how large the mutation rate can become during adaptation.
    /// A maximum rate prevents excessive randomness that could disrupt good solutions that have already been found.
    /// </para>
    /// <para><b>For Beginners:</b> This is like limiting how much fertilizer the gardener can add, 
    /// ensuring they don't add so much that it damages healthy plants.
    /// </para>
    /// </remarks>
    private readonly double _maxMutationRate;

    /// <summary>
    /// The minimum allowed crossover rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value sets a lower bound on how small the crossover rate can become during adaptation.
    /// A minimum rate ensures that genetic information is still exchanged between individuals at a baseline level.
    /// </para>
    /// <para><b>For Beginners:</b> This is like ensuring the gardener always cross-pollinates at least 
    /// some plants, even when a particular variety seems promising.
    /// </para>
    /// </remarks>
    private readonly double _minCrossoverRate;

    /// <summary>
    /// The maximum allowed crossover rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value sets an upper bound on how large the crossover rate can become during adaptation.
    /// A maximum rate prevents excessive mixing that could break apart effective gene combinations.
    /// </para>
    /// <para><b>For Beginners:</b> This is like limiting how many plants the gardener cross-pollinates,
    /// ensuring some successful plant varieties maintain their unique characteristics.
    /// </para>
    /// </remarks>
    private readonly double _maxCrossoverRate;

    /// <summary>
    /// The current mutation rate being applied in the algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the actual mutation rate currently in use during evolution.
    /// It is dynamically adjusted based on population diversity and other factors.
    /// </para>
    /// <para><b>For Beginners:</b> This is the amount of fertilizer the gardener is currently using,
    /// which changes based on how the garden is developing.
    /// </para>
    /// </remarks>
    private double _currentMutationRate;

    /// <summary>
    /// The current crossover rate being applied in the algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the actual crossover rate currently in use during evolution.
    /// It is dynamically adjusted based on improvement rates and other factors.
    /// </para>
    /// <para><b>For Beginners:</b> This is how frequently the gardener is currently cross-pollinating plants,
    /// which changes based on whether the garden is improving or has stagnated.
    /// </para>
    /// </remarks>
    private double _currentCrossoverRate;

    /// <summary>
    /// Initializes a new instance of the AdaptiveGeneticAlgorithm class.
    /// </summary>
    /// <param name="modelFactory">Factory function that creates new model instances.</param>
    /// <param name="fitnessCalculator">The calculator used to determine model fitness.</param>
    /// <param name="modelEvaluator">The evaluator used to assess model performance.</param>
    /// <param name="minMutationRate">The minimum mutation rate allowed (default: 0.001).</param>
    /// <param name="maxMutationRate">The maximum mutation rate allowed (default: 0.5).</param>
    /// <param name="minCrossoverRate">The minimum crossover rate allowed (default: 0.4).</param>
    /// <param name="maxCrossoverRate">The maximum crossover rate allowed (default: 0.95).</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the adaptive genetic algorithm with its initial configuration and boundaries.
    /// The algorithm starts with mutation and crossover rates at the midpoint of their respective ranges.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up the gardening rules with minimum and maximum amounts 
    /// for different gardening techniques. The gardener starts with middle-of-the-road approaches
    /// and will adjust over time based on what works best.
    /// </para>
    /// </remarks>
    public AdaptiveGeneticAlgorithm(
        Func<IFullModel<T, TInput, TOutput>> modelFactory,
        IFitnessCalculator<T, TInput, TOutput> fitnessCalculator,
        IModelEvaluator<T, TInput, TOutput> modelEvaluator,
        double minMutationRate = 0.001,
        double maxMutationRate = 0.5,
        double minCrossoverRate = 0.4,
        double maxCrossoverRate = 0.95)
        : base(modelFactory, fitnessCalculator, modelEvaluator)
    {
        _minMutationRate = minMutationRate;
        _maxMutationRate = maxMutationRate;
        _minCrossoverRate = minCrossoverRate;
        _maxCrossoverRate = maxCrossoverRate;

        // Initialize with default rates
        _currentMutationRate = (minMutationRate + maxMutationRate) / 2;
        _currentCrossoverRate = (minCrossoverRate + maxCrossoverRate) / 2;
    }

    /// <summary>
    /// Evolves the population over a specified number of generations.
    /// </summary>
    /// <param name="generations">The number of generations to evolve.</param>
    /// <param name="trainingInput">The input data used for training.</param>
    /// <param name="trainingOutput">The expected output data for training.</param>
    /// <param name="validationInput">Optional validation input data.</param>
    /// <param name="validationOutput">Optional validation output data.</param>
    /// <param name="stopCriteria">Optional function to determine when to stop evolution early.</param>
    /// <returns>Statistics about the evolution process.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base evolution process to initialize the adaptive rates from genetic parameters
    /// before starting the evolution. It then leverages the standard evolution process, with rates being
    /// dynamically adjusted during each generation through the UpdateEvolutionStats method.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the gardener starting a growing season. 
    /// The gardener first sets up their initial plan based on what worked before (genetic parameters),
    /// then grows plants over many cycles (generations), adjusting their approach after each cycle
    /// based on how well the plants are doing.
    /// </para>
    /// </remarks>
    public override GeneticStats<T, TInput, TOutput> Evolve(
        int generations,
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default,
        Func<GeneticStats<T, TInput, TOutput>, bool>? stopCriteria = null)
    {
        // Initialize rates from genetic parameters
        _currentMutationRate = GeneticParams.MutationRate;
        _currentCrossoverRate = GeneticParams.CrossoverRate;

        // Use base Evolve method
        return base.Evolve(generations, trainingInput, trainingOutput, validationInput, validationOutput, stopCriteria);
    }

    /// <summary>
    /// Updates the evolution statistics and adapts genetic parameters based on current performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method extends the base implementation by calling AdaptParameters after updating the statistics.
    /// This is where the adaptive behavior of the algorithm is implemented, adjusting rates based on
    /// the current state of the evolution.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the gardener analyzing how the garden is growing 
    /// after each cycle. The gardener first updates their notes about plant health and growth,
    /// then decides how to adjust their techniques for the next growing cycle.
    /// </para>
    /// </remarks>
    protected override void UpdateEvolutionStats()
    {
        base.UpdateEvolutionStats();

        // Update the parameters based on the diversity and improvement rate
        AdaptParameters();
    }

    /// <summary>
    /// Adapts the mutation and crossover rates based on current diversity and improvement metrics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the adaptive strategy of the algorithm by adjusting:
    /// - Mutation rate based on population diversity (higher when diversity is low, lower when diversity is high)
    /// - Crossover rate based on improvement status (higher during stagnation, lower when improving)
    /// 
    /// The adapted rates are always kept within the minimum and maximum bounds established at initialization.
    /// </para>
    /// <para><b>For Beginners:</b> This is the gardener's decision-making process:
    /// 
    /// - If plants look too similar (low diversity), add more fertilizer (increase mutation)
    /// - If plants already look very different (high diversity), reduce fertilizer (decrease mutation)
    /// - If plants haven't improved in several cycles (stagnation), change cross-pollination strategy (increase crossover)
    /// - If plants are steadily improving, maintain or slightly reduce cross-pollination (decrease crossover)
    /// 
    /// The gardener always stays within reasonable limits for each technique to avoid damaging the garden.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Gets metadata about the algorithm and its current state.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the algorithm.</returns>
    /// <remarks>
    /// <para>
    /// This method extends the base implementation to include adaptive-specific information in the metadata.
    /// It adds details about the algorithm type, description, and the minimum, maximum, and final values
    /// for mutation and crossover rates.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the gardener creating a detailed report about their 
    /// gardening approach. The report includes not just general information about the garden,
    /// but also specific details about their adaptive techniques and the final settings they used
    /// after many cycles of adjustments.
    /// </para>
    /// </remarks>
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