namespace AiDotNet.Helpers;

public static class AdaptiveParametersHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public static void UpdateAdaptiveGeneticParameters(
        ref T currentCrossoverRate,
        ref T currentMutationRate,
        OptimizationStepData<T> currentStepData,
        OptimizationStepData<T> previousStepData,
        GeneticAlgorithmOptimizerOptions options)
    {
        if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            // Improvement: increase crossover rate, decrease mutation rate
            currentCrossoverRate = NumOps.Multiply(currentCrossoverRate, NumOps.FromDouble(options.CrossoverRateIncrease));
            currentMutationRate = NumOps.Multiply(currentMutationRate, NumOps.FromDouble(options.MutationRateDecay));
        }
        else
        {
            // No improvement: decrease crossover rate, increase mutation rate
            currentCrossoverRate = NumOps.Multiply(currentCrossoverRate, NumOps.FromDouble(options.CrossoverRateDecay));
            currentMutationRate = NumOps.Multiply(currentMutationRate, NumOps.FromDouble(options.MutationRateIncrease));
        }

        // Ensure rates stay within bounds
        currentCrossoverRate = MathHelper.Max(NumOps.FromDouble(options.MinCrossoverRate),
            MathHelper.Min(NumOps.FromDouble(options.MaxCrossoverRate), currentCrossoverRate));

        currentMutationRate = MathHelper.Max(NumOps.FromDouble(options.MinMutationRate),
            MathHelper.Min(NumOps.FromDouble(options.MaxMutationRate), currentMutationRate));
    }
}