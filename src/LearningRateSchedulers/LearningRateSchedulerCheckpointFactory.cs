namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Enforces the portable checkpoint contract for learning-rate schedulers.
/// </summary>
/// <remarks>
/// Restore uses the matching explicit allow-list in <c>GradientBasedOptimizerBase</c>. Keeping
/// capture validation here prevents writing a checkpoint whose immutable recipe cannot be rebuilt.
/// </remarks>
internal static class LearningRateSchedulerCheckpointFactory
{
    /// <summary>
    /// Captures a complete scheduler recipe and mutable progress after validating portability.
    /// </summary>
    internal static Dictionary<string, object> CaptureState(ILearningRateScheduler scheduler)
    {
        Guard.NotNull(scheduler);
        EnsureBuiltInScheduler(scheduler);
        return scheduler.GetState();
    }

    private static void EnsureBuiltInScheduler(ILearningRateScheduler scheduler)
    {
        Type type = scheduler.GetType();
        bool supported =
            type == typeof(ConstantLRScheduler) ||
            type == typeof(StepLRScheduler) ||
            type == typeof(MultiStepLRScheduler) ||
            type == typeof(ExponentialLRScheduler) ||
            type == typeof(PolynomialLRScheduler) ||
            type == typeof(CosineAnnealingLRScheduler) ||
            type == typeof(CosineAnnealingWarmRestartsScheduler) ||
            type == typeof(OneCycleLRScheduler) ||
            type == typeof(LinearWarmupScheduler) ||
            type == typeof(CyclicLRScheduler) ||
            type == typeof(ReduceOnPlateauScheduler) ||
            type == typeof(AdaptiveFitnessScheduler) ||
            type == typeof(NoamSchedule) ||
            type == typeof(SequentialLRScheduler);

        if (!supported)
        {
            string detail = type == typeof(LambdaLRScheduler)
                ? "Lambda scheduler delegates do not have a serializable reconstruction recipe."
                : "Only built-in schedulers with explicit reconstruction recipes are checkpointable.";
            throw new NotSupportedException(
                $"Learning-rate scheduler '{type.FullName}' cannot be checkpointed. {detail}");
        }

        if (scheduler is SequentialLRScheduler sequential)
        {
            foreach (ILearningRateScheduler child in sequential.Schedulers)
            {
                EnsureBuiltInScheduler(child);
            }
        }
    }
}
