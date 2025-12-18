using AiDotNet.Enums;
using AiDotNet.Configuration;
using AiDotNet.Evaluation;
using AiDotNet.Exceptions;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.AutoML.Policies;

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
    /// Gets or sets options controlling optional post-search ensembling.
    /// </summary>
    /// <remarks>
    /// This is primarily used by the facade options overload in <c>PredictionModelBuilder</c>.
    /// </remarks>
    public AutoMLEnsembleOptions EnsembleOptions { get; set; } = new();

    /// <summary>
    /// Gets or sets the compute budget preset used to choose sensible built-in defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Built-in AutoML defaults (for example, candidate model sets) can vary by budget preset so CI runs remain fast
    /// while thorough runs consider a broader model catalog.
    /// </para>
    /// <para><b>For Beginners:</b> A budget preset is like choosing how much time/effort AutoML should spend searching:
    /// CI is very fast, Standard is balanced, and Thorough tries more options.</para>
    /// </remarks>
    public AutoMLBudgetPreset BudgetPreset { get; set; } = AutoMLBudgetPreset.Standard;

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
        catch (AiDotNetException ex)
        {
            var duration = DateTime.UtcNow - trialStart;
            await ReportTrialFailureAsync(trialParameters, ex, duration);
            return _maximize ? double.NegativeInfinity : double.PositiveInfinity;
        }
        catch (ArgumentException ex)
        {
            var duration = DateTime.UtcNow - trialStart;
            await ReportTrialFailureAsync(trialParameters, ex, duration);
            return _maximize ? double.NegativeInfinity : double.PositiveInfinity;
        }
        catch (InvalidOperationException ex)
        {
            var duration = DateTime.UtcNow - trialStart;
            await ReportTrialFailureAsync(trialParameters, ex, duration);
            return _maximize ? double.NegativeInfinity : double.PositiveInfinity;
        }
        catch (NotSupportedException ex)
        {
            var duration = DateTime.UtcNow - trialStart;
            await ReportTrialFailureAsync(trialParameters, ex, duration);
            return _maximize ? double.NegativeInfinity : double.PositiveInfinity;
        }
        catch (ArithmeticException ex)
        {
            var duration = DateTime.UtcNow - trialStart;
            await ReportTrialFailureAsync(trialParameters, ex, duration);
            return _maximize ? double.NegativeInfinity : double.PositiveInfinity;
        }
    }

    /// <summary>
    /// Attempts to build and select an ensemble as the final model based on <see cref="EnsembleOptions"/>.
    /// </summary>
    protected async Task TrySelectEnsembleAsBestAsync(
        TInput trainInputs,
        TOutput trainTargets,
        TInput validationInputs,
        TOutput validationTargets,
        DateTime deadlineUtc,
        CancellationToken cancellationToken)
    {
        try
        {
            if (!EnsembleOptions.Enabled || EnsembleOptions.MaxModelCount < 2)
            {
                return;
            }

            if (DateTime.UtcNow >= deadlineUtc)
            {
                return;
            }

            if (typeof(TInput) != typeof(Matrix<T>) || typeof(TOutput) != typeof(Vector<T>))
            {
                return;
            }

            List<TrialResult> candidates;
            lock (_lock)
            {
                candidates = _trialHistory
                    .Where(t => t.Success && t.Parameters.ContainsKey("ModelType"))
                    .Select(t => t.Clone())
                    .ToList();
            }

            // Multi-fidelity trials may include reduced-budget runs. Ensemble should only consider full-fidelity trials.
            candidates = candidates
                .Where(t =>
                {
                    if (!t.Parameters.TryGetValue("FidelityFraction", out var ff))
                    {
                        return true;
                    }

                    return ff is double fraction && fraction >= 1.0 - 1e-12;
                })
                .ToList();

            if (candidates.Count < 2)
            {
                return;
            }

            var topTrials = candidates
                .OrderByDescending(t => _maximize ? t.Score : -t.Score)
                .Take(Math.Min(EnsembleOptions.MaxModelCount, candidates.Count))
                .ToList();

            var members = new List<IFullModel<T, Matrix<T>, Vector<T>>>();
            var memberScores = new List<double>();

            foreach (var trial in topTrials)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (DateTime.UtcNow >= deadlineUtc)
                {
                    break;
                }

                if (!trial.Parameters.TryGetValue("ModelType", out var modelTypeObj) || modelTypeObj is not ModelType modelType)
                {
                    continue;
                }

                IFullModel<T, TInput, TOutput> model;
                try
                {
                    model = await CreateModelAsync(modelType, trial.Parameters);
                }
                catch (InvalidOperationException)
                {
                    continue;
                }
                catch (ArgumentException)
                {
                    continue;
                }
                catch (NotSupportedException)
                {
                    continue;
                }

                model.Train(trainInputs, trainTargets);

                double score;
                try
                {
                    score = await EvaluateModelAsync(model, validationInputs, validationTargets);
                }
                catch (InvalidOperationException)
                {
                    continue;
                }
                catch (ArgumentException)
                {
                    continue;
                }
                catch (NotSupportedException)
                {
                    continue;
                }

                members.Add((IFullModel<T, Matrix<T>, Vector<T>>)(object)model);
                memberScores.Add(score);
            }

            if (members.Count < 2)
            {
                return;
            }

            if (trainTargets is not Vector<T> targetVector)
            {
                return;
            }

            var predictionType = PredictionTypeInference.Infer(targetVector);
            var weights = ComputeEnsembleWeights(memberScores, _maximize);
            var ensemble = new AutoMLEnsembleModel<T>(members, predictionType, weights);

            double ensembleScore;
            try
            {
                ensembleScore = await EvaluateModelAsync((IFullModel<T, TInput, TOutput>)(object)ensemble, validationInputs, validationTargets);
            }
            catch (InvalidOperationException)
            {
                return;
            }
            catch (ArgumentException)
            {
                return;
            }
            catch (NotSupportedException)
            {
                return;
            }

            bool useEnsemble = EnsembleOptions.FinalSelectionPolicy switch
            {
                AutoMLFinalModelSelectionPolicy.AlwaysUseEnsemble => true,
                AutoMLFinalModelSelectionPolicy.UseEnsembleIfBetter => _maximize ? ensembleScore > BestScore : ensembleScore < BestScore,
                _ => false
            };

            if (useEnsemble)
            {
                BestModel = (IFullModel<T, TInput, TOutput>)(object)ensemble;
                BestScore = ensembleScore;
            }
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception)
        {
            // Ensembling is best-effort; if it fails, keep the best single model.
        }
    }

    private static double[] ComputeEnsembleWeights(IReadOnlyList<double> scores, bool maximize)
    {
        if (scores.Count == 0)
        {
            return Array.Empty<double>();
        }

        var rewards = scores.Select(s => maximize ? s : -s).ToArray();
        double min = rewards.Min();

        // Shift into a positive domain to avoid negative/zero weights.
        for (int i = 0; i < rewards.Length; i++)
        {
            rewards[i] = Math.Max(0.0, rewards[i] - min + 1e-9);
        }

        double sum = rewards.Sum();
        if (sum <= 0 || double.IsNaN(sum) || double.IsInfinity(sum))
        {
            double uniform = 1.0 / rewards.Length;
            return Enumerable.Repeat(uniform, rewards.Length).ToArray();
        }

        for (int i = 0; i < rewards.Length; i++)
        {
            rewards[i] /= sum;
        }

        return rewards;
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

        var taskFamily = AutoMLTaskFamilyInference.InferFromTargets<T, TOutput>(targets);
        var (metric, maximize) = AutoMLDefaultMetricPolicy.GetDefault(taskFamily);
        SetOptimizationMetric(metric, maximize);

        _optimizationMetricExplicitlySet = false;
    }
}
