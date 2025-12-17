using AiDotNet.Enums;
using AiDotNet.Evaluation;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Regression;

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
public class RandomSearchAutoML<T, TInput, TOutput> : SupervisedAutoMLModelBase<T, TInput, TOutput>
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

    protected override Task<IFullModel<T, TInput, TOutput>> CreateModelAsync(ModelType modelType, Dictionary<string, object> parameters)
    {
        if (typeof(TInput) != typeof(Matrix<T>) || typeof(TOutput) != typeof(Vector<T>))
        {
            throw new NotSupportedException(
                $"RandomSearchAutoML currently supports Matrix<T>/Vector<T> supervised tasks. Received {typeof(TInput).Name}/{typeof(TOutput).Name}.");
        }

        IFullModel<T, Matrix<T>, Vector<T>> model = modelType switch
        {
            ModelType.SimpleRegression => CreateWithOptions(
                (RegressionOptions<T> options) => new SimpleRegression<T>(options),
                new RegressionOptions<T>(),
                parameters),

            ModelType.MultipleRegression => CreateWithOptions(
                (RegressionOptions<T> options) => new MultipleRegression<T>(options),
                new RegressionOptions<T>(),
                parameters),

            ModelType.PolynomialRegression => CreateWithOptions(
                (PolynomialRegressionOptions<T> options) => new PolynomialRegression<T>(options),
                new PolynomialRegressionOptions<T>(),
                parameters),

            ModelType.LogisticRegression => CreateWithOptions(
                (LogisticRegressionOptions<T> options) => new LogisticRegression<T>(options),
                new LogisticRegressionOptions<T>(),
                parameters),

            ModelType.MultinomialLogisticRegression => CreateWithOptions(
                (MultinomialLogisticRegressionOptions<T> options) => new MultinomialLogisticRegression<T>(options),
                new MultinomialLogisticRegressionOptions<T>(),
                parameters),

            ModelType.RandomForest => CreateWithOptions(
                (RandomForestRegressionOptions options) => new RandomForestRegression<T>(options),
                new RandomForestRegressionOptions(),
                parameters),

            ModelType.GradientBoosting => CreateWithOptions(
                (GradientBoostingRegressionOptions options) => new GradientBoostingRegression<T>(options),
                new GradientBoostingRegressionOptions(),
                parameters),

            ModelType.KNearestNeighbors => CreateWithOptions(
                (KNearestNeighborsOptions options) => new KNearestNeighborsRegression<T>(options),
                new KNearestNeighborsOptions(),
                parameters),

            ModelType.SupportVectorRegression => CreateWithOptions(
                (SupportVectorRegressionOptions options) => new SupportVectorRegression<T>(options),
                new SupportVectorRegressionOptions(),
                parameters),

            ModelType.NeuralNetworkRegression => new NeuralNetworkRegression<T>(),

            _ => throw new NotSupportedException($"AutoML model type '{modelType}' is not currently supported by RandomSearchAutoML.")
        };

        object boxedModel = model;
        return Task.FromResult((IFullModel<T, TInput, TOutput>)boxedModel);
    }

    protected override Dictionary<string, ParameterRange> GetDefaultSearchSpace(ModelType modelType)
    {
        return modelType switch
        {
            ModelType.PolynomialRegression => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
            {
                ["Degree"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 1,
                    MaxValue = 6,
                    Step = 1,
                    DefaultValue = 2
                }
            },

            ModelType.LogisticRegression => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
            {
                ["MaxIterations"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 100,
                    MaxValue = 5000,
                    Step = 100,
                    DefaultValue = 1000
                },
                ["LearningRate"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 1e-4,
                    MaxValue = 0.1,
                    UseLogScale = true,
                    DefaultValue = 0.01
                },
                ["Tolerance"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 1e-6,
                    MaxValue = 1e-2,
                    UseLogScale = true,
                    DefaultValue = 1e-4
                }
            },

            ModelType.MultinomialLogisticRegression => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
            {
                ["MaxIterations"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 100,
                    MaxValue = 5000,
                    Step = 100,
                    DefaultValue = 1000
                },
                ["LearningRate"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 1e-4,
                    MaxValue = 0.1,
                    UseLogScale = true,
                    DefaultValue = 0.01
                },
                ["Tolerance"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 1e-6,
                    MaxValue = 1e-2,
                    UseLogScale = true,
                    DefaultValue = 1e-4
                }
            },

            ModelType.RandomForest => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
            {
                ["NumberOfTrees"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 50,
                    MaxValue = 500,
                    Step = 10,
                    DefaultValue = 100
                },
                ["MaxDepth"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 2,
                    MaxValue = 50,
                    Step = 1,
                    DefaultValue = 10
                },
                ["MinSamplesSplit"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 2,
                    MaxValue = 50,
                    Step = 1,
                    DefaultValue = 2
                },
                ["MaxFeatures"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 0.2,
                    MaxValue = 1.0,
                    Step = 0.05,
                    DefaultValue = 1.0
                }
            },

            ModelType.GradientBoosting => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
            {
                ["NumberOfTrees"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 50,
                    MaxValue = 500,
                    Step = 10,
                    DefaultValue = 100
                },
                ["LearningRate"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 0.01,
                    MaxValue = 0.3,
                    UseLogScale = true,
                    DefaultValue = 0.1
                },
                ["SubsampleRatio"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 0.5,
                    MaxValue = 1.0,
                    Step = 0.05,
                    DefaultValue = 1.0
                },
                ["MaxDepth"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 2,
                    MaxValue = 20,
                    Step = 1,
                    DefaultValue = 10
                }
            },

            ModelType.KNearestNeighbors => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
            {
                ["K"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 1,
                    MaxValue = 50,
                    Step = 1,
                    DefaultValue = 5
                }
            },

            ModelType.SupportVectorRegression => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
            {
                ["C"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 0.1,
                    MaxValue = 100.0,
                    UseLogScale = true,
                    DefaultValue = 1.0
                },
                ["Epsilon"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 0.001,
                    MaxValue = 1.0,
                    UseLogScale = true,
                    DefaultValue = 0.1
                }
            },

            _ => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        };
    }

    protected override AutoMLModelBase<T, TInput, TOutput> CreateInstanceForCopy()
    {
        return new RandomSearchAutoML<T, TInput, TOutput>(_modelEvaluator, Random);
    }

    private void EnsureDefaultCandidateModels(TInput inputs, TOutput targets)
    {
        lock (_lock)
        {
            if (_candidateModels.Count != 0)
            {
                return;
            }

            if (typeof(TInput) == typeof(Matrix<T>) && typeof(TOutput) == typeof(Vector<T>))
            {
                int featureCount = InputHelper<T, TInput>.GetInputSize(inputs);
                var predictionType = PredictionTypeInference.Infer(ConversionsHelper.ConvertToVector<T, TOutput>(targets));
                bool isClassification =
                    predictionType == PredictionType.Binary
                    || predictionType == PredictionType.MultiClass
                    || predictionType == PredictionType.MultiLabel;

                if (featureCount == 1)
                {
                    _candidateModels.Add(ModelType.SimpleRegression);
                }

                _candidateModels.Add(ModelType.MultipleRegression);
                _candidateModels.Add(ModelType.PolynomialRegression);
                _candidateModels.Add(ModelType.RandomForest);
                _candidateModels.Add(ModelType.GradientBoosting);
                _candidateModels.Add(ModelType.KNearestNeighbors);
                _candidateModels.Add(ModelType.SupportVectorRegression);
                _candidateModels.Add(ModelType.NeuralNetworkRegression);

                if (isClassification)
                {
                    _candidateModels.Add(ModelType.LogisticRegression);
                    _candidateModels.Add(ModelType.MultinomialLogisticRegression);
                }
            }
        }
    }

    private static TModel CreateWithOptions<TOptions, TModel>(
        Func<TOptions, TModel> factory,
        TOptions options,
        IReadOnlyDictionary<string, object> parameters)
    {
        if (factory is null)
        {
            throw new ArgumentNullException(nameof(factory));
        }

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        AutoMLHyperparameterApplicator.ApplyToOptions(options, parameters);
        return factory(options);
    }
}
