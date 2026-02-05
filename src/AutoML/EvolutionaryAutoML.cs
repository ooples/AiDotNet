using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;

namespace AiDotNet.AutoML;

/// <summary>
/// Built-in AutoML strategy that uses an evolutionary (genetic) approach to propose new trials.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This strategy treats each trial configuration as an "individual" and iteratively improves by:
/// - selecting strong prior trials as parents
/// - combining their settings (crossover)
/// - randomly tweaking some settings (mutation)
/// </para>
/// <para>
/// <b>For Beginners:</b> This is like natural selection: keep the best settings, mix them, and make small random changes
/// to discover even better settings over time.
/// </para>
/// </remarks>
public sealed class EvolutionaryAutoML<T, TInput, TOutput> : BuiltInSupervisedAutoMLModelBase<T, TInput, TOutput>
{
    private const int MinSuccessfulTrialsForEvolution = 10;
    private const double EliteQuantile = 0.25;
    private const double MutationRate = 0.2;
    private const double ExplorationRate = 0.1;

    public EvolutionaryAutoML(Random? random = null)
        : base(random)
    {
    }

    public override async Task<IFullModel<T, TInput, TOutput>> SearchAsync(
        TInput inputs,
        TOutput targets,
        TInput validationInputs,
        TOutput validationTargets,
        TimeSpan timeLimit,
        CancellationToken cancellationToken = default)
    {
        Status = AutoMLStatus.Running;
        TimeLimit = timeLimit;

        try
        {
            EnsureDefaultOptimizationMetric(targets);
            EnsureDefaultCandidateModels(inputs, targets);

            var deadline = DateTime.UtcNow.Add(timeLimit);
            int trialCount = 0;

            while (DateTime.UtcNow < deadline && trialCount < TrialLimit)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (ShouldStop())
                {
                    break;
                }

                var parameters = await SuggestNextTrialAsync();

                if (!parameters.TryGetValue("ModelType", out var modelTypeObj) || modelTypeObj is not ModelType modelType)
                {
                    throw new InvalidOperationException("AutoML trial parameters must include a ModelType entry.");
                }

                _ = await ExecuteTrialAsync(modelType, parameters, inputs, targets, validationInputs, validationTargets, cancellationToken);
                trialCount++;
            }

            if (BestModel is null)
            {
                throw new InvalidOperationException("AutoML failed to find a valid model configuration within the given budget.");
            }

            await TrySelectEnsembleAsBestAsync(inputs, targets, validationInputs, validationTargets, deadline, cancellationToken);

            Status = AutoMLStatus.Completed;
            return BestModel;
        }
        catch (OperationCanceledException)
        {
            Status = AutoMLStatus.Cancelled;
            throw;
        }
        catch (Exception)
        {
            Status = AutoMLStatus.Failed;
            throw;
        }
    }

    public override Task<Dictionary<string, object>> SuggestNextTrialAsync()
    {
        var modelType = PickCandidateModelType();
        if (modelType == ModelType.None)
        {
            throw new InvalidOperationException("No candidate models are configured for AutoML.");
        }

        Dictionary<string, ParameterRange> merged;
        List<TrialResult> successfulForType;

        lock (_lock)
        {
            merged = GetDefaultSearchSpace(modelType)
                .ToDictionary(kvp => kvp.Key, kvp => (ParameterRange)kvp.Value.Clone(), StringComparer.Ordinal);

            foreach (var (key, value) in _searchSpace)
            {
                merged[key] = (ParameterRange)value.Clone();
            }

            successfulForType = _trialHistory
                .Where(t => t.Success && t.Parameters.TryGetValue("ModelType", out var mt) && mt is ModelType m && m == modelType)
                .Select(t => t.Clone())
                .ToList();
        }

        Dictionary<string, object> sampled;

        if (successfulForType.Count < MinSuccessfulTrialsForEvolution || Random.NextDouble() < ExplorationRate)
        {
            sampled = AutoMLParameterSampler.Sample(Random, merged);
        }
        else
        {
            sampled = ProposeByEvolution(merged, successfulForType);
        }

        sampled["ModelType"] = modelType;
        return Task.FromResult(sampled);
    }

    protected override AutoMLModelBase<T, TInput, TOutput> CreateInstanceForCopy()
    {
        return new EvolutionaryAutoML<T, TInput, TOutput>(Random);
    }

    private double ToReward(double score) => _maximize ? score : -score;

    private Dictionary<string, object> ProposeByEvolution(
        IReadOnlyDictionary<string, ParameterRange> searchSpace,
        IReadOnlyList<TrialResult> successfulTrials)
    {
        var elite = SelectElite(successfulTrials);
        if (elite.Count == 0)
        {
            return AutoMLParameterSampler.Sample(Random, searchSpace);
        }

        var parentA = TournamentSelect(elite);
        var parentB = TournamentSelect(elite);

        var child = new Dictionary<string, object>(StringComparer.Ordinal);

        foreach (var (name, range) in searchSpace)
        {
            object value;

            bool hasA = parentA.Parameters.TryGetValue(name, out var aVal);
            bool hasB = parentB.Parameters.TryGetValue(name, out var bVal);

            if (hasA && hasB)
            {
                value = Random.NextDouble() < 0.5 ? aVal! : bVal!;
            }
            else if (hasA)
            {
                value = aVal!;
            }
            else if (hasB)
            {
                value = bVal!;
            }
            else
            {
                value = AutoMLParameterSampler.Sample(Random, new Dictionary<string, ParameterRange>(StringComparer.Ordinal) { [name] = range })[name];
            }

            if (Random.NextDouble() < MutationRate)
            {
                value = Mutate(value, range);
            }

            child[name] = ClampToRange(value, range);
        }

        return child;
    }

    private List<TrialResult> SelectElite(IReadOnlyList<TrialResult> successfulTrials)
    {
        var sorted = successfulTrials
            .OrderByDescending(t => ToReward(t.Score))
            .ToList();

        int eliteCount = Math.Max(1, (int)Math.Round(sorted.Count * EliteQuantile));
        return sorted.Take(eliteCount).ToList();
    }

    private TrialResult TournamentSelect(IReadOnlyList<TrialResult> elite)
    {
        if (elite.Count == 1)
        {
            return elite[0];
        }

        int a = Random.Next(elite.Count);
        int b = Random.Next(elite.Count);
        var t1 = elite[a];
        var t2 = elite[b];
        return ToReward(t1.Score) >= ToReward(t2.Score) ? t1 : t2;
    }

    private object Mutate(object current, ParameterRange range)
    {
        switch (range.Type)
        {
            case ParameterType.Boolean:
                return !(current is bool b && b);

            case ParameterType.Categorical:
                if (range.CategoricalValues is null || range.CategoricalValues.Count == 0)
                {
                    return range.DefaultValue ?? string.Empty;
                }

                return range.CategoricalValues[Random.Next(range.CategoricalValues.Count)];

            case ParameterType.Integer:
                return MutateNumeric(current, range, integer: true);

            case ParameterType.Float:
            case ParameterType.Continuous:
                return MutateNumeric(current, range, integer: false);

            default:
                return AutoMLParameterSampler.Sample(Random, new Dictionary<string, ParameterRange>(StringComparer.Ordinal) { ["x"] = range })["x"];
        }
    }

    private object MutateNumeric(object current, ParameterRange range, bool integer)
    {
        double min = range.MinValue is null ? 0.0 : Convert.ToDouble(range.MinValue);
        double max = range.MaxValue is null ? min + 1.0 : Convert.ToDouble(range.MaxValue);
        if (max < min)
        {
            (min, max) = (max, min);
        }

        double value;
        try
        {
            value = Convert.ToDouble(current);
        }
        catch (Exception)
        {
            value = min;
        }

        double span = Math.Max(1e-12, max - min);
        double sigma = 0.15 * span;
        double mutated = value + (sigma * NextGaussian());
        mutated = Math.Max(min, Math.Min(max, mutated));

        if (range.Step.HasValue && range.Step.Value > 0)
        {
            double step = range.Step.Value;
            double steps = Math.Round((mutated - min) / step);
            mutated = min + (steps * step);
            mutated = Math.Max(min, Math.Min(max, mutated));
        }

        return integer ? (object)Convert.ToInt32(Math.Round(mutated)) : mutated;
    }

    private object ClampToRange(object value, ParameterRange range)
    {
        return range.Type switch
        {
            ParameterType.Integer => ClampNumeric(value, range, integer: true),
            ParameterType.Float or ParameterType.Continuous => ClampNumeric(value, range, integer: false),
            _ => value
        };
    }

    private object ClampNumeric(object value, ParameterRange range, bool integer)
    {
        double min = range.MinValue is null ? 0.0 : Convert.ToDouble(range.MinValue);
        double max = range.MaxValue is null ? min + 1.0 : Convert.ToDouble(range.MaxValue);
        if (max < min)
        {
            (min, max) = (max, min);
        }

        double v;
        try
        {
            v = Convert.ToDouble(value);
        }
        catch (Exception)
        {
            v = min;
        }

        v = Math.Max(min, Math.Min(max, v));

        if (range.Step.HasValue && range.Step.Value > 0)
        {
            double step = range.Step.Value;
            double steps = Math.Round((v - min) / step);
            v = min + (steps * step);
            v = Math.Max(min, Math.Min(max, v));
        }

        return integer ? (object)Convert.ToInt32(Math.Round(v)) : v;
    }

    private double NextGaussian()
    {
        return Random.NextGaussian();
    }
}
