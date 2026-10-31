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
    /// <summary>Initializes a new instance with the DifferentialEvolution defaults.</summary>
    public DifferentialEvolutionOptions()
    {
        // Storn & Price DE/rand/1/bin defaults: crossover probability CR = 0.5, differential weight
        // F = 0.8 (carried by MutationRate), population 50.
        CrossoverRate = 0.5;
        MutationRate = 0.8;
        PopulationSize = 50;
    }


}
