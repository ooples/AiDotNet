using AiDotNet.Enums;
using AiDotNet.Evaluation;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.AutoML;

/// <summary>
/// Base class for AutoML implementations that train and score supervised models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This base class provides common trial execution logic (create model, train, evaluate, record results)
/// for AutoML strategies that operate on supervised learning datasets.
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoML is an automatic "model picker + tuner".
/// A supervised AutoML run:
/// <list type="number">
/// <item><description>Tries a candidate model configuration (a "trial").</description></item>
/// <item><description>Trains it on your training data.</description></item>
/// <item><description>Scores it on validation data using a metric (like RMSE or Accuracy).</description></item>
/// <item><description>Repeats until it finds a strong model or runs out of budget.</description></item>
/// </list>
/// Concrete strategies (random search, Bayesian optimization, etc.) decide how to pick the next trial.
/// </para>
/// </remarks>
public abstract class SupervisedAutoMLModelBase<T, TInput, TOutput> : AutoMLModelBase<T, TInput, TOutput>
{
    private readonly Random _random;

    /// <summary>
    /// Initializes a new supervised AutoML model with sensible default dependencies.
    /// </summary>
    /// <param name="modelEvaluator">Optional evaluator; if null, a default evaluator is used.</param>
    /// <param name="random">Optional RNG; if null, a secure RNG is used.</param>
    protected SupervisedAutoMLModelBase(IModelEvaluator<T, TInput, TOutput>? modelEvaluator = null, Random? random = null)
    {
        _random = random ?? RandomHelper.CreateSecureRandom();
        _modelEvaluator = modelEvaluator ?? new DefaultModelEvaluator<T, TInput, TOutput>();
    }

    /// <summary>
    /// Gets the RNG used for sampling candidate trials.
    /// </summary>
    protected Random Random => _random;

    /// <summary>
    /// Runs a single trial (create, train, evaluate, record history).
    /// </summary>
    protected async Task<double> ExecuteTrialAsync(
        ModelType modelType,
        Dictionary<string, object> trialParameters,
        TInput trainInputs,
        TOutput trainTargets,
        TInput validationInputs,
        TOutput validationTargets,
        CancellationToken cancellationToken)
    {
        var trialStart = DateTime.UtcNow;

        try
        {
            cancellationToken.ThrowIfCancellationRequested();

            var model = await CreateModelAsync(modelType, trialParameters);

            cancellationToken.ThrowIfCancellationRequested();
            model.Train(trainInputs, trainTargets);

            cancellationToken.ThrowIfCancellationRequested();
            var score = await EvaluateModelAsync(model, validationInputs, validationTargets);

            var duration = DateTime.UtcNow - trialStart;
            var previousBest = BestScore;

            await ReportTrialResultAsync(trialParameters, score, duration);

            bool improved = _maximize ? score > previousBest : score < previousBest;
            if (improved)
            {
                BestModel = model;
            }

            return score;
        }
        catch (OperationCanceledException)
        {
            Status = AutoMLStatus.Cancelled;
            throw;
        }
        catch (Exception ex)
        {
            var duration = DateTime.UtcNow - trialStart;
            await ReportTrialFailureAsync(trialParameters, ex, duration);
            return _maximize ? double.NegativeInfinity : double.PositiveInfinity;
        }
    }

    /// <summary>
    /// Picks a model type uniformly from the configured candidate list.
    /// </summary>
    protected ModelType PickCandidateModelType()
    {
        lock (_lock)
        {
            if (_candidateModels.Count == 0)
            {
                return ModelType.None;
            }

            return _candidateModels[_random.Next(_candidateModels.Count)];
        }
    }

    /// <summary>
    /// Applies an industry-default metric if the user didn't explicitly choose one.
    /// </summary>
    protected void EnsureDefaultOptimizationMetric(TOutput targets)
    {
        if (_optimizationMetricExplicitlySet)
        {
            return;
        }

        var inferredPredictionType = PredictionTypeInference.Infer(ConversionsHelper.ConvertToVector<T, TOutput>(targets));
        switch (inferredPredictionType)
        {
            case PredictionType.Binary:
            case PredictionType.MultiClass:
            case PredictionType.MultiLabel:
                SetOptimizationMetric(MetricType.Accuracy, maximize: true);
                break;

            default:
                SetOptimizationMetric(MetricType.RMSE, maximize: false);
                break;
        }

        _optimizationMetricExplicitlySet = false;
    }
}
