namespace AiDotNet.Statistics;

/// <summary>
/// Represents statistics about the evolutionary process.
/// </summary>
public class EvolutionStats<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the current generation number.
    /// </summary>
    public int Generation { get; set; }

    /// <summary>
    /// Gets or sets the best fitness found so far.
    /// </summary>
    public T BestFitness { get; set; }

    /// <summary>
    /// Gets or sets the average fitness of the current population.
    /// </summary>
    public T AverageFitness { get; set; }

    /// <summary>
    /// Gets or sets the worst fitness in the current population.
    /// </summary>
    public T WorstFitness { get; set; }

    /// <summary>
    /// Gets or sets a measure of the population's genetic diversity.
    /// </summary>
    public T Diversity { get; set; }

    /// <summary>
    /// Gets or sets the standard deviation of fitness in the population.
    /// </summary>
    public T FitnessStandardDeviation { get; set; }

    /// <summary>
    /// Gets or sets the time elapsed since the evolution started.
    /// </summary>
    public TimeSpan TimeElapsed { get; set; }

    /// <summary>
    /// Gets or sets a reference to the best individual found so far.
    /// </summary>
    public object BestIndividual { get; set; }

    /// <summary>
    /// Gets or sets the fitness history across generations.
    /// </summary>
    public List<T> FitnessHistory { get; set; }

    /// <summary>
    /// Gets or sets whether a fitness improvement occurred in the last generation.
    /// </summary>
    public bool ImprovedInLastGeneration { get; set; }

    /// <summary>
    /// Gets or sets the number of generations since the last improvement.
    /// </summary>
    public int GenerationsSinceImprovement { get; set; }

    public EvolutionStats(IFitnessCalculator<T, TInput, TOutput> fitnessCalculator)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        BestFitness = fitnessCalculator.IsHigherScoreBetter ? numOps.MinValue : numOps.MaxValue;
        WorstFitness = fitnessCalculator.IsHigherScoreBetter ? numOps.MaxValue : numOps.MinValue;
        AverageFitness = numOps.Zero;
        BestIndividual = new();
        FitnessHistory = new List<T>();
        Diversity = numOps.Zero;
        FitnessStandardDeviation = numOps.Zero;
    }
}
