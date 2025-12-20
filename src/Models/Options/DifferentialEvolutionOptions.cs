namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Differential Evolution optimization, a powerful variant of genetic algorithms
/// that is particularly effective for continuous optimization problems.
/// </summary>
/// <remarks>
/// <para>
/// Differential Evolution is a population-based optimization algorithm that uses vector differences for
/// mutation operations. It's known for its robustness, simplicity, and effectiveness in solving complex
/// optimization problems, especially those with real-valued parameters.
/// </para>
/// <para><b>For Beginners:</b> Differential Evolution is like a more sophisticated version of genetic algorithms.
/// Instead of random mutations, it creates new solutions by calculating the difference between existing solutions
/// and using that difference to guide the search. Think of it as learning from the "distance" between good solutions
/// to find even better ones. It's particularly good at finding optimal values for problems with continuous variables
/// (like finding the best temperature, pressure, or other numeric values).</para>
/// </remarks>
public class DifferentialEvolutionOptions<T, TInput, TOutput> : GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the crossover rate for Differential Evolution.
    /// </summary>
    /// <value>The crossover rate, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// In Differential Evolution, the crossover rate determines the probability of accepting components from
    /// the mutated vector rather than the original vector when creating trial solutions. A value of 0.5 means
    /// there's an equal chance of selecting components from either vector.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much mixing happens between the original solution and the
    /// newly created one. With the default value of 0.5, there's a 50% chance that each part of the new solution
    /// will come from the modified version rather than the original. Unlike regular genetic algorithms, in
    /// Differential Evolution this mixing happens between an existing solution and its modified version, not
    /// between two different solutions.</para>
    /// </remarks>
    public new double CrossoverRate { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the mutation rate (also known as the differential weight or F) for Differential Evolution.
    /// </summary>
    /// <value>The mutation rate, defaulting to 0.8.</value>
    /// <remarks>
    /// <para>
    /// In Differential Evolution, the mutation rate (F) scales the difference vector when creating mutant vectors.
    /// It controls the amplification of the differential variation. Values between 0.5 and 1.0 are commonly used,
    /// with 0.8 being a good general-purpose value.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how strongly the algorithm uses the differences between existing
    /// solutions to create new ones. With the default value of 0.8, the algorithm takes 80% of the difference
    /// between solutions when creating variations. Think of it as determining how bold or conservative the algorithm
    /// is when exploring new possibilities. Higher values make bigger jumps in the solution space, while lower
    /// values make smaller, more careful adjustments.</para>
    /// </remarks>
    public new double MutationRate { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the size of the population for Differential Evolution.
    /// </summary>
    /// <value>The population size, defaulting to 50.</value>
    /// <remarks>
    /// <para>
    /// The population size determines how many individual solutions are maintained in each generation.
    /// For Differential Evolution, smaller population sizes (like 50) are often sufficient compared to
    /// traditional genetic algorithms, as the algorithm is more efficient at exploring the solution space.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many different solutions the algorithm works with at once.
    /// Differential Evolution typically needs fewer solutions than regular genetic algorithms to find good answers,
    /// which is why the default is 50 instead of 100. Having the right number of solutions helps balance between
    /// finding good answers (which needs diversity) and computing efficiently (which favors fewer solutions).</para>
    /// </remarks>
    public new int PopulationSize { get; set; } = 50;
}
