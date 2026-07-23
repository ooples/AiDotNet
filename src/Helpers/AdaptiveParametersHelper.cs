namespace AiDotNet.Helpers;

/// <summary>
/// Helper class that provides methods for dynamically adjusting genetic algorithm parameters during optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This helper class contains methods that automatically adjust the settings of a genetic algorithm
/// while it's running to help it find better solutions.
/// 
/// A genetic algorithm is an AI technique inspired by natural evolution - it creates a "population" of possible 
/// solutions, selects the best ones, and combines them to create new solutions, similar to how animals evolve 
/// through natural selection.
/// 
/// Two important settings in genetic algorithms are:
/// - Crossover rate: How often solutions are combined to create new ones (like breeding)
/// - Mutation rate: How often random changes are introduced to solutions (like genetic mutations)
/// 
/// This class helps the algorithm perform better by automatically adjusting these rates based on whether
/// the algorithm is making progress or getting stuck.
/// </para>
/// </remarks>
public static class AdaptiveParametersHelper<T, TInput, TOutput>
{
    /// <summary>
    /// Provides numeric operations appropriate for the generic type T.
    /// </summary>
    /// <remarks>
    /// This field allows the helper to perform mathematical operations regardless of the numeric type used.
    /// </remarks>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Updates the crossover and mutation rates based on whether the optimization is improving.
    /// </summary>
    /// <param name="currentCrossoverRate">The current crossover rate to be updated.</param>
    /// <param name="currentMutationRate">The current mutation rate to be updated.</param>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <param name="options">Configuration options for the genetic algorithm optimizer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is like an automatic tuning system for the genetic algorithm.
    /// 
    /// Imagine you're trying to find the best recipe for a cake. If your latest attempt tastes better than 
    /// the previous one, you might want to:
    /// - Make more small variations of your successful recipe (increase crossover)
    /// - Make fewer random changes (decrease mutation)
    /// 
    /// But if your latest cake tastes worse, you might want to:
    /// - Try more different approaches (decrease crossover)
    /// - Make more random changes to escape the bad recipe (increase mutation)
    /// 
    /// That's exactly what this method does for the genetic algorithm:
    /// 
    /// 1. It compares the current result (fitness score) with the previous one
    /// 2. If things are improving, it adjusts the settings to focus more on refining good solutions
    /// 3. If things are getting worse, it adjusts the settings to explore more diverse solutions
    /// 4. It always makes sure the settings stay within reasonable limits
    /// 
    /// This adaptive approach helps the algorithm work better across different problems without
    /// requiring manual tuning.
    /// </para>
    /// </remarks>
    public static void UpdateAdaptiveGeneticParameters(
        ref T currentCrossoverRate,
        ref T currentMutationRate,
        OptimizationStepData<T, TInput, TOutput> currentStepData,
        OptimizationStepData<T, TInput, TOutput> previousStepData,
        GeneticAlgorithmOptimizerOptions<T, TInput, TOutput> options)
    {
        if (_numOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            // Improvement: increase crossover rate, decrease mutation rate
            currentCrossoverRate = _numOps.Multiply(currentCrossoverRate, _numOps.FromDouble(options.CrossoverRateIncrease));
            currentMutationRate = _numOps.Multiply(currentMutationRate, _numOps.FromDouble(options.MutationRateDecay));
        }
        else
        {
            // No improvement: decrease crossover rate, increase mutation rate
            currentCrossoverRate = _numOps.Multiply(currentCrossoverRate, _numOps.FromDouble(options.CrossoverRateDecay));
            currentMutationRate = _numOps.Multiply(currentMutationRate, _numOps.FromDouble(options.MutationRateIncrease));
        }

        // Ensure rates stay within bounds
        currentCrossoverRate = MathHelper.Max(_numOps.FromDouble(options.MinCrossoverRate),
            MathHelper.Min(_numOps.FromDouble(options.MaxCrossoverRate), currentCrossoverRate));

        currentMutationRate = MathHelper.Max(_numOps.FromDouble(options.MinMutationRate),
            MathHelper.Min(_numOps.FromDouble(options.MaxMutationRate), currentMutationRate));
    }
}
