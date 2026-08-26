using System.Globalization;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Enforces the portable checkpoint contract for learning-rate schedulers.
/// </summary>
/// <remarks>
/// The factory table is the single source of truth for both checkpoint capture and restore.
/// A scheduler cannot be admitted for serialization unless this class also owns its complete,
/// allow-listed reconstruction recipe.
/// </remarks>
internal static class LearningRateSchedulerCheckpointFactory
{
    private static readonly IReadOnlyDictionary<Type, Func<Dictionary<string, object>, ILearningRateScheduler>>
        BuiltInFactories = new Dictionary<Type, Func<Dictionary<string, object>, ILearningRateScheduler>>
        {
            [typeof(ConstantLRScheduler)] = state =>
                new ConstantLRScheduler(StateValue<double>(state, "base_lr")),
            [typeof(StepLRScheduler)] = state =>
                new StepLRScheduler(
                    StateValue<double>(state, "base_lr"),
                    StateValue<int>(state, "step_size"),
                    StateValue<double>(state, "gamma"),
                    StateValue<double>(state, "min_lr")),
            [typeof(ExponentialLRScheduler)] = state =>
                new ExponentialLRScheduler(
                    StateValue<double>(state, "base_lr"),
                    StateValue<double>(state, "gamma"),
                    StateValue<double>(state, "min_lr")),
            [typeof(CosineAnnealingLRScheduler)] = state =>
                new CosineAnnealingLRScheduler(
                    StateValue<double>(state, "base_lr"),
                    StateValue<int>(state, "t_max"),
                    StateValue<double>(state, "eta_min")),
            [typeof(CosineAnnealingWarmRestartsScheduler)] = state =>
                new CosineAnnealingWarmRestartsScheduler(
                    StateValue<double>(state, "base_lr"),
                    StateValue<int>(state, "t0"),
                    StateValue<int>(state, "t_mult"),
                    StateValue<double>(state, "eta_min")),
            [typeof(CyclicLRScheduler)] = state =>
                new CyclicLRScheduler(
                    StateValue<double>(state, "base_lr"),
                    StateValue<double>(state, "max_learning_rate"),
                    StateValue<int>(state, "step_size_up"),
                    StateValue<int>(state, "step_size_down"),
                    StateEnum<CyclicLRScheduler.CyclicMode>(state, "mode"),
                    StateValue<double>(state, "gamma")),
            [typeof(LinearWarmupScheduler)] = state =>
                new LinearWarmupScheduler(
                    StateValue<double>(state, "base_lr"),
                    StateValue<int>(state, "warmup_steps"),
                    StateValue<int>(state, "total_steps"),
                    StateValue<double>(state, "warmup_init_lr"),
                    StateEnum<LinearWarmupScheduler.DecayMode>(state, "decay_mode"),
                    StateValue<double>(state, "end_lr")),
            [typeof(MultiStepLRScheduler)] = state =>
                new MultiStepLRScheduler(
                    StateValue<double>(state, "base_lr"),
                    StateValue<int[]>(state, "milestones"),
                    StateValue<double>(state, "gamma"),
                    StateValue<double>(state, "min_lr")),
            [typeof(NoamSchedule)] = state =>
                new NoamSchedule(
                    StateValue<int>(state, "model_dimension"),
                    StateValue<int>(state, "warmup_steps"),
                    StateValue<double>(state, "factor")),
            [typeof(OneCycleLRScheduler)] = state =>
                new OneCycleLRScheduler(
                    StateValue<double>(state, "max_learning_rate"),
                    StateValue<int>(state, "total_steps"),
                    StateValue<double>(state, "pct_start"),
                    StateValue<double>(state, "div_factor"),
                    StateValue<double>(state, "final_div_factor"),
                    StateEnum<OneCycleLRScheduler.AnnealingStrategy>(state, "anneal_strategy")),
            [typeof(PolynomialLRScheduler)] = state =>
                new PolynomialLRScheduler(
                    StateValue<double>(state, "base_lr"),
                    StateValue<int>(state, "total_steps"),
                    StateValue<double>(state, "power"),
                    StateValue<double>(state, "end_lr")),
            [typeof(ReduceOnPlateauScheduler)] = state =>
                new ReduceOnPlateauScheduler(
                    StateValue<double>(state, "base_lr"),
                    StateValue<double>(state, "factor"),
                    StateValue<int>(state, "patience"),
                    StateValue<double>(state, "threshold"),
                    StateEnum<ReduceOnPlateauScheduler.ThresholdMode>(state, "threshold_mode"),
                    StateValue<int>(state, "cooldown"),
                    StateEnum<ReduceOnPlateauScheduler.Mode>(state, "mode"),
                    StateValue<double>(state, "min_lr")),
            [typeof(AdaptiveFitnessScheduler)] = state =>
                new AdaptiveFitnessScheduler(
                    StateValue<double>(state, "base_lr"),
                    StateValue<double>(state, "decay"),
                    StateValue<double>(state, "min_lr"),
                    StateValue<double>(state, "max_learning_rate"),
                    StateValue<bool>(state, "higher_is_better")),
            [typeof(SequentialLRScheduler)] = CreateSequentialScheduler,
        };

    private static readonly IReadOnlyDictionary<string, Func<Dictionary<string, object>, ILearningRateScheduler>>
        BuiltInFactoriesByTypeName = BuiltInFactories.ToDictionary(
            pair => pair.Key.FullName!,
            pair => pair.Value,
            StringComparer.Ordinal);

    /// <summary>
    /// Captures a complete scheduler recipe and mutable progress after validating portability.
    /// </summary>
    internal static Dictionary<string, object> CaptureState(ILearningRateScheduler scheduler)
    {
        Guard.NotNull(scheduler);
        EnsureBuiltInScheduler(scheduler);
        return scheduler.GetState();
    }

    /// <summary>
    /// Reconstructs an allow-listed built-in scheduler without loading a checkpoint-supplied type.
    /// </summary>
    internal static bool TryCreateBuiltInScheduler(
        string serializedTypeName,
        Dictionary<string, object> state,
        out ILearningRateScheduler scheduler)
    {
        if (!BuiltInFactoriesByTypeName.TryGetValue(serializedTypeName, out var factory))
        {
            scheduler = null!;
            return false;
        }

        scheduler = factory(state);
        return true;
    }

    private static void EnsureBuiltInScheduler(ILearningRateScheduler scheduler)
    {
        Type type = scheduler.GetType();
        if (!BuiltInFactories.ContainsKey(type))
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

    private static ILearningRateScheduler CreateSequentialScheduler(Dictionary<string, object> state)
    {
        var schedulerTypes = StateValue<string[]>(state, "scheduler_types");
        var schedulerStates = StateValue<Dictionary<string, object>[]>(state, "scheduler_states");
        if (schedulerTypes.Length == 0 || schedulerTypes.Length != schedulerStates.Length)
        {
            throw new InvalidOperationException(
                "Sequential scheduler checkpoint has mismatched child type and state counts.");
        }

        var children = new List<ILearningRateScheduler>(schedulerTypes.Length);
        for (int i = 0; i < schedulerTypes.Length; i++)
        {
            string childTypeName = schedulerTypes[i].Split(',')[0].Trim();
            if (!TryCreateBuiltInScheduler(childTypeName, schedulerStates[i], out var child))
            {
                throw new InvalidOperationException(
                    $"Sequential scheduler child type '{childTypeName}' is not a checkpoint-supported built-in.");
            }

            children.Add(child);
        }

        return new SequentialLRScheduler(children, StateValue<int[]>(state, "milestones"));
    }

    private static TState StateValue<TState>(Dictionary<string, object> state, string key)
    {
        if (!state.TryGetValue(key, out var value) || value is null)
        {
            throw new InvalidOperationException(
                $"Learning-rate scheduler checkpoint is missing required recipe value '{key}'.");
        }

        try
        {
            if (value is TState typedValue)
                return typedValue;
            if (value is JToken token)
            {
                return token.ToObject<TState>()
                    ?? throw new InvalidOperationException(
                        $"Learning-rate scheduler recipe value '{key}' cannot be converted to {typeof(TState).Name}.");
            }

            return (TState)Convert.ChangeType(value, typeof(TState), CultureInfo.InvariantCulture);
        }
        catch (Exception ex) when (ex is FormatException or InvalidCastException or OverflowException or JsonException)
        {
            throw new InvalidOperationException(
                $"Learning-rate scheduler recipe value '{key}' is invalid.", ex);
        }
    }

    private static TEnum StateEnum<TEnum>(Dictionary<string, object> state, string key)
        where TEnum : struct, Enum
    {
        string value = StateValue<string>(state, key);
        if (!Enum.TryParse<TEnum>(value, ignoreCase: false, out var parsed)
            || !Enum.IsDefined(typeof(TEnum), parsed))
        {
            throw new InvalidOperationException(
                $"Learning-rate scheduler recipe value '{key}' has unsupported {typeof(TEnum).Name} value '{value}'.");
        }

        return parsed;
    }
}
