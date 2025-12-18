using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.AutoML;

/// <summary>
/// AutoML implementation that uses random search over candidate model types and hyperparameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Random search is a strong baseline for AutoML. It is simple, parallelizable, and often competitive with
/// more complex search strategies for a given compute budget.
/// </para>
/// <para>
/// <b>For Beginners:</b> This AutoML strategy works like this:
/// <list type="number">
/// <item><description>Pick a model type at random (for example, Random Forest or Logistic Regression).</description></item>
/// <item><description>Pick a set of settings at random (for example, number of trees).</description></item>
/// <item><description>Train the model and score it on validation data.</description></item>
/// <item><description>Repeat and keep the best result.</description></item>
/// </list>
/// If you are new to AutoML, random search is a good first choice because it is reliable and easy to reason about.
/// </para>
/// </remarks>
public class RandomSearchAutoML<T, TInput, TOutput> : BuiltInSupervisedAutoMLModelBase<T, TInput, TOutput>
{
    public RandomSearchAutoML(IModelEvaluator<T, TInput, TOutput>? modelEvaluator = null, Random? random = null)
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
                var history = GetTrialHistory();
                int totalTrials = history.Count;
                int failedTrials = history.Count(t => !t.Success);
                int successfulTrials = totalTrials - failedTrials;

                var errorSamples = history
                    .Where(t => !t.Success && !string.IsNullOrWhiteSpace(t.ErrorMessage))
                    .Select(t => t.ErrorMessage!)
                    .Distinct(StringComparer.Ordinal)
                    .Take(3)
                    .ToArray();

                var details = errorSamples.Length == 0
                    ? $"Trials: {totalTrials} (success: {successfulTrials}, failed: {failedTrials})."
                    : $"Trials: {totalTrials} (success: {successfulTrials}, failed: {failedTrials}). Sample errors: {string.Join(" | ", errorSamples)}";

                throw new InvalidOperationException($"AutoML failed to find a valid model configuration within the given budget. {details}");
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
        lock (_lock)
        {
            merged = GetDefaultSearchSpace(modelType)
                .ToDictionary(kvp => kvp.Key, kvp => (ParameterRange)kvp.Value.Clone(), StringComparer.Ordinal);

            foreach (var (key, value) in _searchSpace)
            {
                merged[key] = (ParameterRange)value.Clone();
            }
        }

        var sampled = AutoMLParameterSampler.Sample(Random, merged);
        sampled["ModelType"] = modelType;
        return Task.FromResult(sampled);
    }

    protected override AutoMLModelBase<T, TInput, TOutput> CreateInstanceForCopy()
    {
        return new RandomSearchAutoML<T, TInput, TOutput>(_modelEvaluator, Random);
    }
}
