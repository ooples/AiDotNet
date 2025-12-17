namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for genetic algorithm optimization, which uses principles inspired by natural selection
/// to find optimal solutions to complex problems.
/// </summary>
/// <remarks>
/// <para>
/// Genetic algorithms simulate the process of natural selection where the fittest individuals are selected for
/// reproduction to produce offspring for the next generation. This approach is particularly effective for
/// optimization problems with large search spaces or complex constraints.
/// </para>
/// <para><b>For Beginners:</b> A genetic algorithm works like breeding animals or plants to get desired traits.
/// You start with a diverse group of potential solutions (the "population"), evaluate how good each one is,
/// let the best ones "reproduce" by combining their characteristics, occasionally introduce random changes
/// ("mutations"), and repeat this process over multiple "generations" until you find an excellent solution.
/// It's a way to solve problems by mimicking how nature evolves species over time.</para>
/// </remarks>
public class GeneticAlgorithmOptimizerOptions<T, TInput, TOutput> : OptimizationAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the maximum number of generations (iterations) the genetic algorithm will run.
    /// </summary>
    /// <value>The maximum number of generations, defaulting to 50.</value>
    /// <remarks>
    /// <para>
    /// This value limits how many evolutionary cycles the algorithm will perform before stopping,
    /// even if an optimal solution hasn't been found. It prevents excessive computation time.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting a time limit for the evolution process.
    /// With the default value of 50, the algorithm will create up to 50 generations of solutions
    /// before stopping. Think of it as deciding how many rounds of breeding you'll try before
    /// picking the best result you've found so far. More generations give better results but
    /// take longer to compute.</para>
    /// </remarks>
    public int MaxGenerations { get; set; } = 50;

    /// <summary>
    /// Gets or sets the size of the population for genetic and evolutionary algorithms.
    /// </summary>
    /// <value>The population size, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// The population size determines how many individual solutions are maintained in each generation.
    /// A larger population provides more genetic diversity but requires more computational resources.
    /// </para>
    /// <para><b>For Beginners:</b> In genetic algorithms, a "population" is a collection of potential solutions.
    /// This setting controls how many different solutions the algorithm works with at once. A larger population
    /// provides more diversity but requires more computational resources. Think of it like having more people
    /// brainstorming solutions to a problem.</para>
    /// </remarks>
    public int PopulationSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets the minimum allowed population size for genetic algorithms.
    /// </summary>
    /// <value>The minimum population size, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// This value sets a lower bound on how small the population can become during adaptive adjustments.
    /// It prevents the algorithm from losing genetic diversity due to extremely small populations.
    /// </para>
    /// <para><b>For Beginners:</b> This ensures the algorithm always works with at least this many different
    /// solutions at once. Having too few solutions can limit diversity and make it harder to find good answers.
    /// Think of it as making sure you have enough different ideas to consider.</para>
    /// </remarks>
    public int MinPopulationSize { get; set; } = 10;

    /// <summary>
    /// Gets or sets the maximum allowed population size for genetic algorithms.
    /// </summary>
    /// <value>The maximum population size, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This value sets an upper bound on how large the population can become during adaptive adjustments.
    /// It prevents excessive computational resource usage from extremely large populations.
    /// </para>
    /// <para><b>For Beginners:</b> This limits how many different solutions the algorithm works with at once.
    /// Having too many solutions can slow down the algorithm without providing much benefit. Think of it as
    /// avoiding having so many ideas that you can't properly evaluate them all.</para>
    /// </remarks>
    public int MaxPopulationSize { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the rate at which solutions are combined to create new ones in genetic algorithms.
    /// </summary>
    /// <value>The crossover rate, defaulting to 0.7.</value>
    /// <remarks>
    /// <para>
    /// The crossover rate determines the probability that two parent solutions will exchange genetic material
    /// to produce offspring. Higher rates promote more exploration of the solution space.
    /// </para>
    /// <para><b>For Beginners:</b> Crossover is like breeding two good solutions to create a new one that
    /// hopefully has the best qualities of both parents. A rate of 0.7 means there's a 70% chance that
    /// crossover will occur between two selected solutions. Higher rates encourage more mixing of solutions.</para>
    /// </remarks>
    public double CrossoverRate { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the mutation rate for genetic and evolutionary algorithms.
    /// </summary>
    /// <value>The mutation rate, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// The mutation rate determines the probability that each gene (component) of a solution will be
    /// randomly altered. Mutations help maintain genetic diversity and prevent premature convergence.
    /// </para>
    /// <para><b>For Beginners:</b> In genetic algorithms, mutation introduces random changes to solutions.
    /// A rate of 0.01 means there's a 1% chance of each part of a solution being randomly changed.
    /// This helps discover new possibilities that might not be found otherwise.</para>
    /// </remarks>
    public double MutationRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the factor by which the crossover rate decreases when progress stalls or reverses.
    /// </summary>
    /// <value>The crossover rate decay factor, defaulting to 0.95 (5% decrease).</value>
    /// <remarks>
    /// <para>
    /// When the optimization encounters difficulties or starts to converge, the crossover rate is multiplied
    /// by this factor to focus more on exploiting existing good solutions rather than exploring new ones.
    /// A value less than 1.0 ensures the crossover rate decreases.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the algorithm reduces the amount of solution mixing
    /// (crossover) when progress slows down. With the default value of 0.95, the crossover rate decreases by 5%
    /// when the algorithm isn't finding better solutions. It's like focusing more on refining the best ideas you
    /// already have rather than trying completely new combinations when your current approach is working well.</para>
    /// </remarks>
    public double CrossoverRateDecay { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the factor by which the crossover rate increases when progress is being made.
    /// </summary>
    /// <value>The crossover rate increase factor, defaulting to 1.05 (5% increase).</value>
    /// <remarks>
    /// <para>
    /// When the optimization is making good progress, the crossover rate is multiplied by this factor
    /// to encourage more exploration of the solution space. A value greater than 1.0 ensures the crossover rate increases.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the algorithm increases the amount of solution mixing
    /// (crossover) when it's making good progress. With the default value of 1.05, the crossover rate increases by 5%
    /// when better solutions are being found. It's like encouraging more experimentation with new combinations
    /// when your current approach is yielding improvements.</para>
    /// </remarks>
    public double CrossoverRateIncrease { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the minimum allowed crossover rate for genetic algorithms.
    /// </summary>
    /// <value>The minimum crossover rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This value sets a lower bound on how small the crossover rate can become during adaptive adjustments.
    /// It ensures that some level of genetic recombination always occurs, maintaining diversity.
    /// </para>
    /// <para><b>For Beginners:</b> This ensures that solutions are combined (crossover) at least some of the time.
    /// With the default value of 0.1, there's at least a 10% chance that two selected solutions will be combined.
    /// This helps ensure that good qualities from different solutions can be mixed together.</para>
    /// </remarks>
    public double MinCrossoverRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum allowed crossover rate for genetic algorithms.
    /// </summary>
    /// <value>The maximum crossover rate, defaulting to 0.9.</value>
    /// <remarks>
    /// <para>
    /// This value sets an upper bound on how large the crossover rate can become during adaptive adjustments.
    /// It ensures that some solutions are preserved without recombination, maintaining good genetic material.
    /// </para>
    /// <para><b>For Beginners:</b> This limits how often solutions are combined (crossover). With the default
    /// value of 0.9, there's at most a 90% chance that two selected solutions will be combined. This ensures
    /// that some solutions are carried forward unchanged, preserving good qualities that might be lost in combining.</para>
    /// </remarks>
    public double MaxCrossoverRate { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the factor by which the mutation rate decreases when progress is being made.
    /// </summary>
    /// <value>The mutation rate decay factor, defaulting to 0.95 (5% decrease).</value>
    /// <remarks>
    /// <para>
    /// When the optimization is making good progress, the mutation rate is multiplied by this factor
    /// to focus more on exploiting existing good solutions rather than introducing random changes.
    /// A value less than 1.0 ensures the mutation rate decreases.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the algorithm reduces random changes (mutations)
    /// when it's making good progress. With the default value of 0.95, the mutation rate decreases by 5%
    /// when better solutions are being found. It's like being more careful about changing things when your
    /// current approach is working well - you make fewer random adjustments to avoid disrupting success.</para>
    /// </remarks>
    public double MutationRateDecay { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the factor by which the mutation rate increases when progress stalls or reverses.
    /// </summary>
    /// <value>The mutation rate increase factor, defaulting to 1.05 (5% increase).</value>
    /// <remarks>
    /// <para>
    /// When the optimization encounters difficulties or starts to converge prematurely, the mutation rate is multiplied
    /// by this factor to encourage more exploration of the solution space. A value greater than 1.0 ensures the 
    /// mutation rate increases.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the algorithm increases random changes (mutations)
    /// when progress slows down. With the default value of 1.05, the mutation rate increases by 5%
    /// when the algorithm isn't finding better solutions. It's like shaking things up when you're stuck -
    /// introducing more randomness to help discover new possibilities that might lead to better solutions.
    /// Think of it as trying more experimental approaches when your current strategy isn't working well.</para>
    /// </remarks>
    public double MutationRateIncrease { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the minimum allowed mutation rate for genetic algorithms.
    /// </summary>
    /// <value>The minimum mutation rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para>
    /// This value sets a lower bound on how small the mutation rate can become during adaptive adjustments.
    /// It ensures that some level of genetic mutation always occurs, preventing complete stagnation.
    /// </para>
    /// <para><b>For Beginners:</b> This ensures that at least some random changes (mutations) always occur,
    /// even if the algorithm is performing well. With the default value of 0.001, there's at least a 0.1%
    /// chance of each part of a solution being randomly changed. This helps maintain diversity and prevents
    /// the algorithm from getting stuck.</para>
    /// </remarks>
    public double MinMutationRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the maximum allowed mutation rate for genetic algorithms.
    /// </summary>
    /// <value>The maximum mutation rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This value sets an upper bound on how large the mutation rate can become during adaptive adjustments.
    /// It prevents excessive randomness that could disrupt the optimization process.
    /// </para>
    /// <para><b>For Beginners:</b> This limits how many random changes (mutations) can occur. With the default
    /// value of 0.1, there's at most a 10% chance of each part of a solution being randomly changed. This
    /// prevents too much randomness from disrupting good solutions that have already been found.</para>
    /// </remarks>
    public double MaxMutationRate { get; set; } = 0.1;
}
