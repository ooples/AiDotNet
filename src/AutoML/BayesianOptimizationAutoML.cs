using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;

namespace AiDotNet.AutoML;

/// <summary>
/// Built-in AutoML strategy that uses a lightweight Bayesian-style surrogate to guide trial selection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This implementation uses a pragmatic, production-friendly approach:
/// - Use a bandit policy to allocate trials across candidate model families.
/// - Use a kernel-weighted surrogate over observed trials to bias sampling toward promising regions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of trying totally random settings every time, this strategy learns from earlier
/// trials and tries more settings similar to the best ones found so far.
/// </para>
/// </remarks>
public sealed class BayesianOptimizationAutoML<T, TInput, TOutput> : BuiltInSupervisedAutoMLModelBase<T, TInput, TOutput>
{
    private const int MinSuccessfulTrialsForGuidance = 12;
    private const int CandidateBatchSize = 64;
    private const double KernelScale = 3.0;
    private const double GoodQuantile = 0.2;
    private const double ExplorationRate = 0.15;
    private const double UcbExploration = 0.7;

    public BayesianOptimizationAutoML(IModelEvaluator<T, TInput, TOutput>? modelEvaluator = null, Random? random = null)
        : base(modelEvaluator, random)
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
        var modelType = PickModelTypeByUcb();
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

        var sampled = successfulForType.Count < MinSuccessfulTrialsForGuidance || Random.NextDouble() < ExplorationRate
            ? AutoMLParameterSampler.Sample(Random, merged)
            : SuggestByKernelSurrogate(merged, successfulForType);

        sampled["ModelType"] = modelType;
        return Task.FromResult(sampled);
    }

    protected override AutoMLModelBase<T, TInput, TOutput> CreateInstanceForCopy()
    {
        return new BayesianOptimizationAutoML<T, TInput, TOutput>(_modelEvaluator, Random);
    }

    private ModelType PickModelTypeByUcb()
    {
        lock (_lock)
        {
            if (_candidateModels.Count == 0)
            {
                return ModelType.None;
            }

            var stats = new Dictionary<ModelType, (int Count, double MeanReward)>();
            int total = 0;

            foreach (var modelType in _candidateModels)
            {
                stats[modelType] = (0, 0.0);
            }

            foreach (var trial in _trialHistory)
            {
                if (!trial.Success)
                {
                    continue;
                }

                if (!trial.Parameters.TryGetValue("ModelType", out var mt) || mt is not ModelType modelType)
                {
                    continue;
                }

                if (!stats.ContainsKey(modelType))
                {
                    continue;
                }

                total++;
                var reward = ToReward(trial.Score);
                var (count, mean) = stats[modelType];
                count++;
                mean += (reward - mean) / count;
                stats[modelType] = (count, mean);
            }

            foreach (var modelType in _candidateModels)
            {
                if (stats[modelType].Count == 0)
                {
                    return modelType;
                }
            }

            double logTotal = Math.Log(Math.Max(1, total));
            ModelType best = _candidateModels[0];
            double bestUcb = double.NegativeInfinity;

            foreach (var modelType in _candidateModels)
            {
                var (count, mean) = stats[modelType];
                double ucb = mean + (UcbExploration * Math.Sqrt(logTotal / Math.Max(1, count)));
                if (ucb > bestUcb)
                {
                    bestUcb = ucb;
                    best = modelType;
                }
            }

            return best;
        }
    }

    private double ToReward(double score) => _maximize ? score : -score;

    private Dictionary<string, object> SuggestByKernelSurrogate(
        IReadOnlyDictionary<string, ParameterRange> searchSpace,
        IReadOnlyList<TrialResult> successfulTrials)
    {
        if (successfulTrials.Count == 0)
        {
            return AutoMLParameterSampler.Sample(Random, searchSpace);
        }

        var sorted = successfulTrials
            .OrderByDescending(t => ToReward(t.Score))
            .ToList();

        int goodCount = Math.Max(1, (int)Math.Round(sorted.Count * GoodQuantile));
        var good = sorted.Take(goodCount).ToList();

        Dictionary<string, object>? bestCandidate = null;
        double bestPredicted = double.NegativeInfinity;

        for (int i = 0; i < CandidateBatchSize; i++)
        {
            var candidate = SampleGuidedCandidate(searchSpace, good);
            double predicted = PredictReward(candidate, searchSpace, successfulTrials);

            if (predicted > bestPredicted)
            {
                bestPredicted = predicted;
                bestCandidate = candidate;
            }
        }

        return bestCandidate ?? AutoMLParameterSampler.Sample(Random, searchSpace);
    }

    private Dictionary<string, object> SampleGuidedCandidate(
        IReadOnlyDictionary<string, ParameterRange> searchSpace,
        IReadOnlyList<TrialResult> goodTrials)
    {
        var candidate = new Dictionary<string, object>(StringComparer.Ordinal);

        foreach (var (name, range) in searchSpace)
        {
            if (goodTrials.Count == 0 || Random.NextDouble() < ExplorationRate)
            {
                candidate[name] = AutoMLParameterSampler.Sample(Random, new Dictionary<string, ParameterRange>(StringComparer.Ordinal) { [name] = range })[name];
                continue;
            }

            var values = goodTrials
                .Select(t => t.Parameters.TryGetValue(name, out var v) ? v : null)
                .Where(v => v is not null)
                .ToList();

            if (values.Count == 0)
            {
                candidate[name] = AutoMLParameterSampler.Sample(Random, new Dictionary<string, ParameterRange>(StringComparer.Ordinal) { [name] = range })[name];
                continue;
            }

            candidate[name] = range.Type switch
            {
                ParameterType.Categorical => SampleCategorical(values!, range),
                ParameterType.Boolean => SampleCategorical(values!, range),
                ParameterType.Integer => SampleNumeric(values!, range, integer: true),
                ParameterType.Float or ParameterType.Continuous => SampleNumeric(values!, range, integer: false),
                _ => AutoMLParameterSampler.Sample(Random, new Dictionary<string, ParameterRange>(StringComparer.Ordinal) { [name] = range })[name]
            };
        }

        return candidate;
    }

    private object SampleCategorical(IReadOnlyList<object> values, ParameterRange range)
    {
        var weights = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var v in values)
        {
            var key = v?.ToString() ?? string.Empty;
            weights.TryGetValue(key, out int count);
            weights[key] = count + 1;
        }

        int total = weights.Values.Sum();
        int pick = Random.Next(0, Math.Max(1, total));
        foreach (var (key, count) in weights)
        {
            pick -= count;
            if (pick < 0)
            {
                if (range.Type == ParameterType.Boolean)
                {
                    return bool.TryParse(key, out var b) && b;
                }

                return key;
            }
        }

        return range.DefaultValue ?? string.Empty;
    }

    private object SampleNumeric(IReadOnlyList<object> values, ParameterRange range, bool integer)
    {
        double min = range.MinValue is null ? 0.0 : Convert.ToDouble(range.MinValue);
        double max = range.MaxValue is null ? min + 1.0 : Convert.ToDouble(range.MaxValue);
        if (max < min)
        {
            (min, max) = (max, min);
        }

        if (Math.Abs(max - min) < double.Epsilon)
        {
            return integer ? (object)Convert.ToInt32(min) : min;
        }

        var normalized = new List<double>(values.Count);
        foreach (var v in values)
        {
            try
            {
                double dv = Convert.ToDouble(v);
                normalized.Add(Normalize(dv, range, min, max));
            }
            catch (Exception)
            {
                // ignore malformed values
            }
        }

        if (normalized.Count == 0)
        {
            return AutoMLParameterSampler.Sample(Random, new Dictionary<string, ParameterRange>(StringComparer.Ordinal) { ["x"] = range })["x"];
        }

        double mean = normalized.Average();
        double std = Math.Sqrt(normalized.Sum(x => (x - mean) * (x - mean)) / Math.Max(1, normalized.Count - 1));
        std = Math.Max(0.05, Math.Min(0.25, std));

        double sample = Clamp01(mean + (std * NextGaussian()));
        double value = Denormalize(sample, range, min, max);

        if (range.Step.HasValue && range.Step.Value > 0)
        {
            double step = range.Step.Value;
            double steps = Math.Round((value - min) / step);
            value = min + (steps * step);
            value = Math.Max(min, Math.Min(max, value));
        }

        return integer ? (object)Convert.ToInt32(Math.Round(value)) : value;
    }

    private double PredictReward(
        IReadOnlyDictionary<string, object> candidate,
        IReadOnlyDictionary<string, ParameterRange> searchSpace,
        IReadOnlyList<TrialResult> history)
    {
        double weighted = 0.0;
        double weights = 0.0;
        double fallbackMean = history.Select(t => ToReward(t.Score)).DefaultIfEmpty(0.0).Average();

        foreach (var trial in history)
        {
            double dist = 0.0;
            foreach (var (name, range) in searchSpace)
            {
                if (!candidate.TryGetValue(name, out var cv))
                {
                    continue;
                }

                if (!trial.Parameters.TryGetValue(name, out var tv))
                {
                    continue;
                }

                dist += Distance(cv, tv, range);
            }

            double w = Math.Exp(-KernelScale * dist);
            weighted += w * ToReward(trial.Score);
            weights += w;
        }

        if (weights <= 1e-12)
        {
            return fallbackMean;
        }

        return weighted / weights;
    }

    private static double Distance(object candidateValue, object trialValue, ParameterRange range)
    {
        switch (range.Type)
        {
            case ParameterType.Categorical:
            case ParameterType.Boolean:
                return string.Equals(candidateValue?.ToString(), trialValue?.ToString(), StringComparison.OrdinalIgnoreCase) ? 0.0 : 1.0;

            case ParameterType.Integer:
            case ParameterType.Float:
            case ParameterType.Continuous:
            {
                double min = range.MinValue is null ? 0.0 : Convert.ToDouble(range.MinValue);
                double max = range.MaxValue is null ? min + 1.0 : Convert.ToDouble(range.MaxValue);
                if (max < min)
                {
                    (min, max) = (max, min);
                }

                if (Math.Abs(max - min) < double.Epsilon)
                {
                    return 0.0;
                }

                double c = Convert.ToDouble(candidateValue);
                double t = Convert.ToDouble(trialValue);
                double cn = Normalize(c, range, min, max);
                double tn = Normalize(t, range, min, max);
                return Math.Abs(cn - tn);
            }

            default:
                return 0.0;
        }
    }

    private static double Normalize(double value, ParameterRange range, double min, double max)
    {
        if (range.UseLogScale && min > 0 && max > 0 && value > 0)
        {
            double logMin = Math.Log(min);
            double logMax = Math.Log(max);
            double logVal = Math.Log(value);
            return Clamp01((logVal - logMin) / Math.Max(1e-12, logMax - logMin));
        }

        return Clamp01((value - min) / Math.Max(1e-12, max - min));
    }

    private static double Denormalize(double normalized, ParameterRange range, double min, double max)
    {
        normalized = Clamp01(normalized);

        if (range.UseLogScale && min > 0 && max > 0)
        {
            double logMin = Math.Log(min);
            double logMax = Math.Log(max);
            double logVal = logMin + ((logMax - logMin) * normalized);
            return Math.Exp(logVal);
        }

        return min + ((max - min) * normalized);
    }

    private double NextGaussian()
    {
        return Random.NextGaussian();
    }

    private static double Clamp01(double value) => value < 0 ? 0 : value > 1 ? 1 : value;
}
