using AiDotNet.AutoML;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.Finance.AutoML;

/// <summary>
/// AutoML implementation for finance models (forecasting and risk).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FinancialAutoML searches across a curated set of finance models while preserving
/// the facade pattern. You provide the architecture and budget; AutoML selects the model.
/// </para>
/// <para>
/// <b>For Beginners:</b> This class is a "model picker" for finance tasks.
/// It tries several finance models and chooses the one that scores best on your data.
/// </para>
/// </remarks>
public class FinancialAutoML<T> : SupervisedAutoMLModelBase<T, Tensor<T>, Tensor<T>>
{
    private readonly FinancialAutoMLOptions<T> _options;
    private readonly FinancialSearchSpace _financeSearchSpace;
    private readonly FinancialModelFactory<T> _modelFactory;

    /// <summary>
    /// Initializes a new FinancialAutoML instance.
    /// </summary>
    /// <param name="options">AutoML configuration options.</param>
    /// <param name="random">Optional random number generator.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Provide your architecture and choose a budget.
    /// AutoML will take care of model selection.
    /// </para>
    /// </remarks>
    public FinancialAutoML(
        FinancialAutoMLOptions<T>? options = null,
        Random? random = null)
        : base(random)
    {
        _options = options ?? new FinancialAutoMLOptions<T>();
        _options.Validate();

        if (_options.SearchStrategy != AutoMLSearchStrategy.RandomSearch)
        {
            throw new NotSupportedException(
                $"FinancialAutoML currently supports only '{AutoMLSearchStrategy.RandomSearch}'. " +
                $"Received '{_options.SearchStrategy}'.");
        }

        _financeSearchSpace = new FinancialSearchSpace(_options.Domain);
        _modelFactory = new FinancialModelFactory<T>(_options.Architecture!);

        ApplyBudget(_options.Budget);

        if (_options.CrossValidation is not null)
        {
            CrossValidationOptions = _options.CrossValidation;
        }

        if (_options.OptimizationMetricOverride.HasValue)
        {
            var metric = _options.OptimizationMetricOverride.Value;
            SetOptimizationMetric(metric, maximize: IsHigherBetter(metric));
        }

        if (_options.CandidateModels is not null && _options.CandidateModels.Count > 0)
        {
            SetCandidateModels(_options.CandidateModels);
        }
        else
        {
            SetCandidateModels(GetDefaultModelsForDomain(_options.Domain));
        }
    }

    /// <summary>
    /// Runs the AutoML search loop.
    /// </summary>
    /// <param name="inputs">Training inputs.</param>
    /// <param name="targets">Training targets.</param>
    /// <param name="validationInputs">Validation inputs.</param>
    /// <param name="validationTargets">Validation targets.</param>
    /// <param name="timeLimit">Time limit for the search.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The best model found.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML repeatedly tries candidate models until
    /// it runs out of time or reaches the trial limit.
    /// </para>
    /// </remarks>
    public override async Task<IFullModel<T, Tensor<T>, Tensor<T>>> SearchAsync(
        Tensor<T> inputs,
        Tensor<T> targets,
        Tensor<T> validationInputs,
        Tensor<T> validationTargets,
        TimeSpan timeLimit,
        CancellationToken cancellationToken = default)
    {
        Status = AutoMLStatus.Running;
        TimeLimit = timeLimit;

        try
        {
            EnsureDefaultOptimizationMetric();
            EnsureDefaultCandidateModels();

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
                throw new InvalidOperationException("FinancialAutoML did not find a valid model within the given budget.");
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

    /// <summary>
    /// Suggests the next trial parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This picks a model type and samples any tunable settings.
    /// </para>
    /// </remarks>
    public override Task<Dictionary<string, object>> SuggestNextTrialAsync()
    {
        var modelType = PickCandidateModelType();
        if (modelType == ModelType.None)
        {
            throw new InvalidOperationException("No candidate models are configured for FinancialAutoML.");
        }

        Dictionary<string, ParameterRange> merged;
        lock (_lock)
        {
            merged = _financeSearchSpace.GetSearchSpace(modelType)
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

    /// <summary>
    /// Creates a finance model for the given trial parameters.
    /// </summary>
    /// <param name="modelType">The model type to create.</param>
    /// <param name="parameters">Trial parameters.</param>
    /// <returns>The created model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML builds each candidate model here before training it.
    /// </para>
    /// </remarks>
    protected override Task<IFullModel<T, Tensor<T>, Tensor<T>>> CreateModelAsync(
        ModelType modelType,
        Dictionary<string, object> parameters)
    {
        var model = _modelFactory.Create(modelType, parameters);
        return Task.FromResult(model);
    }

    /// <summary>
    /// Gets the default search space for a model type.
    /// </summary>
    /// <param name="modelType">The model type.</param>
    /// <returns>Search space dictionary.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells AutoML which settings it can tune.
    /// </para>
    /// </remarks>
    protected override Dictionary<string, ParameterRange> GetDefaultSearchSpace(ModelType modelType)
    {
        return _financeSearchSpace.GetSearchSpace(modelType);
    }

    /// <summary>
    /// Creates a new instance for cloning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML uses this to make a copy of itself with the same options.
    /// </para>
    /// </remarks>
    protected override AutoMLModelBase<T, Tensor<T>, Tensor<T>> CreateInstanceForCopy()
    {
        return new FinancialAutoML<T>(_options, Random);
    }

    /// <summary>
    /// Applies the AutoML budget to time and trial limits.
    /// </summary>
    /// <param name="budget">The budget options to apply.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The budget controls how long AutoML is allowed to search.
    /// This method translates preset choices into actual time limits and trial counts.
    /// </para>
    /// </remarks>
    private void ApplyBudget(AutoMLBudgetOptions budget)
    {
        var (defaultTime, defaultTrials) = AutoMLBudgetDefaults.Resolve(budget.Preset);
        TimeLimit = budget.TimeLimitOverride ?? defaultTime;
        TrialLimit = budget.TrialLimitOverride ?? defaultTrials;
        BudgetPreset = budget.Preset;
    }

    /// <summary>
    /// Ensures a default optimization metric is selected when none is specified.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML needs a scoring rule to decide which model is best.
    /// If you do not specify one, this method picks a sensible default for the domain.
    /// </para>
    /// </remarks>
    private void EnsureDefaultOptimizationMetric()
    {
        if (_optimizationMetricExplicitlySet)
        {
            return;
        }

        var metric = _options.Domain == FinancialDomain.Risk
            ? MetricType.MAE
            : MetricType.RMSE;

        SetOptimizationMetric(metric, maximize: IsHigherBetter(metric));
        _optimizationMetricExplicitlySet = false;
    }

    /// <summary>
    /// Ensures a default candidate model list is available.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML needs a list of models to try.
    /// If you do not provide one, this method supplies a curated default list.
    /// </para>
    /// </remarks>
    private void EnsureDefaultCandidateModels()
    {
        lock (_lock)
        {
            if (_candidateModels.Count != 0)
            {
                return;
            }

            foreach (var candidate in GetDefaultModelsForDomain(_options.Domain))
            {
                _candidateModels.Add(candidate);
            }
        }
    }

    /// <summary>
    /// Gets the default candidate models for a finance domain.
    /// </summary>
    /// <param name="domain">The finance domain to target.</param>
    /// <returns>List of model types for the domain.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Forecasting models differ from risk models.
    /// This method picks a small, safe starter set based on the task type.
    /// </para>
    /// </remarks>
    private static List<ModelType> GetDefaultModelsForDomain(FinancialDomain domain)
    {
        return domain switch
        {
            FinancialDomain.Forecasting => new List<ModelType>
            {
                ModelType.PatchTST,
                ModelType.ITransformer,
                ModelType.DeepAR,
                ModelType.NBEATS,
                ModelType.TFT
            },
            FinancialDomain.Risk => new List<ModelType>
            {
                ModelType.NeuralVaR,
                ModelType.TabNet,
                ModelType.TabTransformer
            },
            _ => new List<ModelType>()
        };
    }

    /// <summary>
    /// Determines whether a metric should be maximized or minimized.
    /// </summary>
    /// <param name="metric">The metric to evaluate.</param>
    /// <returns>True if higher is better, otherwise false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Some metrics are errors (lower is better) and some are scores
    /// (higher is better). AutoML needs to know which direction to optimize.
    /// </para>
    /// </remarks>
    private static bool IsHigherBetter(MetricType metric)
    {
        return metric switch
        {
            MetricType.MeanSquaredError => false,
            MetricType.RootMeanSquaredError => false,
            MetricType.MeanAbsoluteError => false,
            MetricType.MSE => false,
            MetricType.RMSE => false,
            MetricType.MAE => false,
            MetricType.MAPE => false,
            MetricType.SMAPE => false,
            MetricType.MeanSquaredLogError => false,
            MetricType.CrossEntropyLoss => false,
            MetricType.AIC => false,
            MetricType.BIC => false,
            MetricType.AICAlt => false,
            MetricType.Perplexity => false,
            _ => true
        };
    }
}
