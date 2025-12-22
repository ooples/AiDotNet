using System.Diagnostics;
using AiDotNet.ActiveLearning.Config;
using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ActiveLearning.Results;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.Core;

/// <summary>
/// Core implementation of the active learner that orchestrates the active learning loop.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input features.</typeparam>
/// <typeparam name="TOutput">The type of output labels.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The ActiveLearner is the main orchestrator that runs the active
/// learning process. It manages the labeled and unlabeled pools, coordinates with the query
/// strategy to select informative samples, and trains the model iteratively.</para>
///
/// <para><b>Active Learning Workflow:</b></para>
/// <list type="number">
/// <item><description>Initialize with labeled and unlabeled data pools</description></item>
/// <item><description>Train the model on labeled data</description></item>
/// <item><description>Use query strategy to select informative unlabeled samples</description></item>
/// <item><description>Get labels from oracle (human expert or simulator)</description></item>
/// <item><description>Move newly labeled samples from unlabeled to labeled pool</description></item>
/// <item><description>Repeat until stopping criterion is met</description></item>
/// </list>
///
/// <para><b>Example Usage:</b></para>
/// <code>
/// var config = new ActiveLearnerConfig&lt;double&gt;
/// {
///     QueryBatchSize = 10,
///     MaxBudget = 500,
///     QueryStrategy = QueryStrategyType.UncertaintySampling
/// };
///
/// var learner = new ActiveLearner&lt;double, double[], int&gt;(model, strategy, config);
/// learner.Initialize(initialLabeled, unlabeledPool);
///
/// var result = learner.Run(oracle, stoppingCriterion);
/// Console.WriteLine($"Final accuracy: {result.FinalTrainingAccuracy}");
/// </code>
/// </remarks>
public class ActiveLearner<T, TInput, TOutput> : IActiveLearner<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly IQueryStrategy<T, TInput, TOutput> _queryStrategy;
    private readonly ActiveLearnerConfig<T> _config;
    private readonly Stopwatch _stopwatch;
    private readonly List<LearningCurvePoint<T>> _learningCurvePoints;
    private readonly Random _random;

    private IDataset<T, TInput, TOutput> _labeledPool;
    private IDataset<T, TInput, TOutput> _unlabeledPool;
    private IDataset<T, TInput, TOutput>? _validationSet;
    private bool _initialized;
    private int _totalQueries;
    private int _iterationsCompleted;
    private TimeSpan _totalTrainingTime;
    private TimeSpan _totalSelectionTime;

    /// <inheritdoc/>
    public ActiveLearnerConfig<T> Configuration => _config;

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> Model => _model;

    /// <inheritdoc/>
    public IQueryStrategy<T, TInput, TOutput> QueryStrategy => _queryStrategy;

    /// <inheritdoc/>
    public IDataset<T, TInput, TOutput> LabeledPool => _labeledPool;

    /// <inheritdoc/>
    public IDataset<T, TInput, TOutput> UnlabeledPool => _unlabeledPool;

    /// <inheritdoc/>
    public int TotalQueries => _totalQueries;

    /// <inheritdoc/>
    public int IterationsCompleted => _iterationsCompleted;

    /// <inheritdoc/>
    public event EventHandler<ActiveLearningIterationResult<T>>? IterationCompleted;

    /// <inheritdoc/>
    public event EventHandler<SamplesSelectedEventArgs<TInput>>? SamplesSelected;

    /// <inheritdoc/>
    public event EventHandler<ActiveLearningResult<T>>? LearningCompleted;

    /// <summary>
    /// Initializes a new ActiveLearner with the specified model, strategy, and configuration.
    /// </summary>
    /// <param name="model">The machine learning model to train.</param>
    /// <param name="queryStrategy">The strategy for selecting samples to query.</param>
    /// <param name="config">Configuration options for active learning.</param>
    public ActiveLearner(
        IFullModel<T, TInput, TOutput> model,
        IQueryStrategy<T, TInput, TOutput> queryStrategy,
        ActiveLearnerConfig<T>? config = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _queryStrategy = queryStrategy ?? throw new ArgumentNullException(nameof(queryStrategy));
        _config = config ?? new ActiveLearnerConfig<T>();

        _stopwatch = new Stopwatch();
        _learningCurvePoints = new List<LearningCurvePoint<T>>();
        _random = _config.Seed.HasValue ? RandomHelper.CreateSeededRandom(_config.Seed.Value) : RandomHelper.Shared;

        _labeledPool = null!;
        _unlabeledPool = null!;
        _initialized = false;
        _totalQueries = 0;
        _iterationsCompleted = 0;
        _totalTrainingTime = TimeSpan.Zero;
        _totalSelectionTime = TimeSpan.Zero;
    }

    /// <inheritdoc/>
    public void Initialize(
        IDataset<T, TInput, TOutput> initialLabeled,
        IDataset<T, TInput, TOutput> unlabeledPool)
    {
        _labeledPool = initialLabeled ?? throw new ArgumentNullException(nameof(initialLabeled));
        _unlabeledPool = unlabeledPool ?? throw new ArgumentNullException(nameof(unlabeledPool));

        // Optionally split off a validation set
        if (_config.EvaluatePerIteration ?? true)
        {
            var testFraction = NumOps.ToDouble(_config.TestSetFraction ?? NumOps.FromDouble(0.2));
            if (testFraction > 0 && testFraction < 1 && _labeledPool.Count > 10)
            {
                var (train, validation) = _labeledPool.Split(1.0 - testFraction, _random);
                _labeledPool = train;
                _validationSet = validation;
            }
        }

        // Initial training on labeled data
        if (_labeledPool.Count > 0)
        {
            TrainModelInternal();
        }

        // Record initial learning curve point
        RecordLearningCurvePoint();

        // Initialize query strategy
        _queryStrategy.Reset();

        _initialized = true;
        _totalQueries = _labeledPool.Count;
        _iterationsCompleted = 0;
    }

    /// <inheritdoc/>
    public ActiveLearningIterationResult<T> RunIteration(IOracle<TInput, TOutput> oracle)
    {
        EnsureInitialized();

        var iterationStopwatch = Stopwatch.StartNew();
        var iterationResult = new ActiveLearningIterationResult<T>
        {
            IterationNumber = _iterationsCompleted
        };

        // Step 1: Select samples to query
        var selectionStopwatch = Stopwatch.StartNew();
        var selectedIndices = SelectNextBatch();
        selectionStopwatch.Stop();
        iterationResult.SelectionTime = selectionStopwatch.Elapsed;
        _totalSelectionTime += selectionStopwatch.Elapsed;

        if (selectedIndices.Length == 0)
        {
            // No more samples to query
            iterationResult.SamplesQueried = 0;
            iterationResult.TotalLabeledSamples = _labeledPool.Count;
            iterationResult.UnlabeledRemaining = _unlabeledPool.Count;
            return iterationResult;
        }

        // Get the selected samples and their scores
        var selectedInputs = new TInput[selectedIndices.Length];
        var scores = new double[selectedIndices.Length];

        for (int i = 0; i < selectedIndices.Length; i++)
        {
            selectedInputs[i] = _unlabeledPool.GetInput(selectedIndices[i]);
        }

        // Compute scores for the selected samples
        var allScores = _queryStrategy.ComputeScores(_model, _unlabeledPool);
        for (int i = 0; i < selectedIndices.Length; i++)
        {
            scores[i] = NumOps.ToDouble(allScores[selectedIndices[i]]);
        }

        // Fire samples selected event
        OnSamplesSelected(new SamplesSelectedEventArgs<TInput>(selectedIndices, selectedInputs, scores));

        // Step 2: Get labels from oracle
        var labels = oracle.LabelBatch(selectedInputs);

        // Step 3: Add labeled samples
        AddLabeledSamples(selectedIndices, labels);

        // Step 4: Train model
        var trainingStopwatch = Stopwatch.StartNew();
        var trainingMetrics = TrainModel();
        trainingStopwatch.Stop();
        iterationResult.TrainingTime = trainingStopwatch.Elapsed;
        _totalTrainingTime += trainingStopwatch.Elapsed;

        // Record learning curve point
        RecordLearningCurvePoint();

        // Update query strategy state
        _queryStrategy.UpdateState(selectedIndices, labels);

        // Populate result
        iterationResult.SamplesQueried = selectedIndices.Length;
        iterationResult.TotalLabeledSamples = _labeledPool.Count;
        iterationResult.TrainingAccuracy = trainingMetrics.Accuracy;
        iterationResult.TrainingLoss = trainingMetrics.Loss;
        iterationResult.QueriedIndices = selectedIndices;
        iterationResult.QueryScores = scores.Select(s => NumOps.FromDouble(s)).ToArray();
        iterationResult.AverageQueryScore = NumOps.FromDouble(scores.Average());
        iterationResult.MaxQueryScore = NumOps.FromDouble(scores.Max());
        iterationResult.UnlabeledRemaining = _unlabeledPool.Count;

        // Validation metrics if available
        if (_validationSet != null && _validationSet.Count > 0)
        {
            var validationMetrics = Evaluate(_validationSet);
            iterationResult.ValidationAccuracy = validationMetrics.Accuracy;
            iterationResult.ValidationLoss = validationMetrics.Loss;
        }

        iterationStopwatch.Stop();
        iterationResult.IterationTime = iterationStopwatch.Elapsed;

        _iterationsCompleted++;

        // Fire iteration completed event
        OnIterationCompleted(iterationResult);

        return iterationResult;
    }

    /// <inheritdoc/>
    public ActiveLearningResult<T> Run(
        IOracle<TInput, TOutput> oracle,
        IStoppingCriterion<T>? stoppingCriterion = null)
    {
        EnsureInitialized();

        _stopwatch.Restart();
        var iterationResults = new List<ActiveLearningIterationResult<T>>();
        var initialLabeled = _labeledPool.Count;

        // Create default stopping criterion if none provided
        stoppingCriterion ??= CreateDefaultStoppingCriterion();

        // Create context for stopping criterion
        var context = CreateLearningContext();

        string stoppingReason = "Maximum iterations reached";

        while (!stoppingCriterion.ShouldStop(context))
        {
            // Check budget
            if (_totalQueries >= _config.GetEffectiveMaxBudget())
            {
                stoppingReason = "Budget exhausted";
                break;
            }

            // Check if unlabeled pool is exhausted
            if (_unlabeledPool.Count == 0)
            {
                stoppingReason = "Unlabeled pool exhausted";
                break;
            }

            // Run one iteration
            var iterationResult = RunIteration(oracle);
            iterationResults.Add(iterationResult);

            // Update context for stopping criterion
            context = CreateLearningContext();

            // Check if iteration made progress
            if (iterationResult.SamplesQueried == 0)
            {
                stoppingReason = "No samples available to query";
                break;
            }
        }

        if (stoppingCriterion.ShouldStop(context))
        {
            stoppingReason = $"Stopping criterion met: {stoppingCriterion.Name}";
        }

        _stopwatch.Stop();

        // Build final result
        var result = BuildFinalResult(iterationResults, initialLabeled, stoppingReason);

        // Fire learning completed event
        OnLearningCompleted(result);

        return result;
    }

    /// <inheritdoc/>
    public int[] SelectNextBatch()
    {
        EnsureInitialized();

        if (_unlabeledPool.Count == 0)
        {
            return Array.Empty<int>();
        }

        int batchSize = Math.Min(
            _config.GetEffectiveQueryBatchSize(),
            _unlabeledPool.Count);

        if (batchSize <= 0)
        {
            return Array.Empty<int>();
        }

        return _queryStrategy.SelectSamples(_model, _unlabeledPool, batchSize);
    }

    /// <inheritdoc/>
    public void AddLabeledSamples(int[] indices, TOutput[] labels)
    {
        EnsureInitialized();

        if (indices.Length != labels.Length)
        {
            throw new ArgumentException("Indices and labels must have the same length.");
        }

        if (indices.Length == 0)
        {
            return;
        }

        // Get inputs for the selected indices
        var inputs = new TInput[indices.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            inputs[i] = _unlabeledPool.GetInput(indices[i]);
        }

        // Add to labeled pool
        _labeledPool = _labeledPool.AddSamples(inputs, labels);

        // Remove from unlabeled pool
        _unlabeledPool = _unlabeledPool.RemoveSamples(indices);

        _totalQueries += indices.Length;
    }

    /// <inheritdoc/>
    public TrainingMetrics<T> TrainModel()
    {
        EnsureInitialized();
        return TrainModelInternal();
    }

    /// <inheritdoc/>
    public EvaluationMetrics<T> Evaluate(IDataset<T, TInput, TOutput> testData)
    {
        if (testData == null || testData.Count == 0)
        {
            return new EvaluationMetrics<T>
            {
                Accuracy = NumOps.Zero,
                Loss = NumOps.Zero
            };
        }

        int correct = 0;
        T totalLoss = NumOps.Zero;

        for (int i = 0; i < testData.Count; i++)
        {
            var input = testData.GetInput(i);
            var expected = testData.GetOutput(i);
            var predicted = _model.Predict(input);

            // Simple accuracy check (for classification)
            if (EqualityComparer<TOutput>.Default.Equals(predicted, expected))
            {
                correct++;
            }

            // Compute loss using model's loss function
            try
            {
                var lossValue = ComputeSampleLoss(predicted, expected);
                totalLoss = NumOps.Add(totalLoss, lossValue);
            }
            catch
            {
                // Loss computation may not be available for all model types
            }
        }

        var accuracy = NumOps.FromDouble((double)correct / testData.Count);
        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(testData.Count));

        return new EvaluationMetrics<T>
        {
            Accuracy = accuracy,
            Loss = avgLoss
        };
    }

    /// <inheritdoc/>
    public LearningCurve<T> GetLearningCurve()
    {
        var curve = new LearningCurve<T>
        {
            SampleCounts = _learningCurvePoints.Select(p => p.SampleCount).ToArray(),
            Accuracies = _learningCurvePoints.Select(p => p.Accuracy).ToArray(),
            Losses = _learningCurvePoints.Select(p => p.Loss).ToArray()
        };

        if (_learningCurvePoints.Any(p => p.ValidationAccuracy != null))
        {
            curve.ValidationAccuracies = _learningCurvePoints
                .Select(p => p.ValidationAccuracy ?? NumOps.Zero)
                .ToArray();
            curve.ValidationLosses = _learningCurvePoints
                .Select(p => p.ValidationLoss ?? NumOps.Zero)
                .ToArray();
        }

        // Compute Area Under Learning Curve (AULC)
        curve.AreaUnderCurve = ComputeAULC(curve);

        return curve;
    }

    #region Private Methods

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException(
                "ActiveLearner must be initialized before use. Call Initialize() first.");
        }
    }

    private TrainingMetrics<T> TrainModelInternal()
    {
        var trainingStopwatch = Stopwatch.StartNew();

        // Train on all labeled data
        for (int i = 0; i < _labeledPool.Count; i++)
        {
            var (input, output) = _labeledPool.GetSample(i);
            _model.Train(input, output);
        }

        trainingStopwatch.Stop();

        // Evaluate on training set
        var trainingEval = Evaluate(_labeledPool);

        return new TrainingMetrics<T>
        {
            Loss = trainingEval.Loss,
            Accuracy = trainingEval.Accuracy,
            EpochsTrained = _config.EpochsPerIteration ?? 10,
            TrainingTime = trainingStopwatch.Elapsed
        };
    }

    private void RecordLearningCurvePoint()
    {
        var trainingEval = Evaluate(_labeledPool);
        var point = new LearningCurvePoint<T>(NumOps)
        {
            SampleCount = _labeledPool.Count,
            Accuracy = trainingEval.Accuracy,
            Loss = trainingEval.Loss
        };

        if (_validationSet != null && _validationSet.Count > 0)
        {
            var validationEval = Evaluate(_validationSet);
            point.ValidationAccuracy = validationEval.Accuracy;
            point.ValidationLoss = validationEval.Loss;
        }

        _learningCurvePoints.Add(point);
    }

    private ActiveLearningContext<T> CreateLearningContext()
    {
        var queryScoreHistory = new List<T>();
        foreach (var point in _learningCurvePoints)
        {
            // Use accuracy as a proxy for query score trend
            queryScoreHistory.Add(point.Accuracy);
        }

        return new ActiveLearningContext<T>
        {
            TotalLabeled = _labeledPool.Count,
            UnlabeledRemaining = _unlabeledPool.Count,
            MaxBudget = _config.GetEffectiveMaxBudget(),
            CurrentIteration = _iterationsCompleted,
            ElapsedTime = _stopwatch.Elapsed,
            MaxTime = null,
            AccuracyHistory = _learningCurvePoints.Select(p => p.Accuracy).ToList(),
            QueryScoreHistory = queryScoreHistory
        };
    }

    private IStoppingCriterion<T> CreateDefaultStoppingCriterion()
    {
        // Default: stop when budget is exhausted or pool is empty
        var budgetCriterion = new StoppingCriteria.BudgetExhaustedCriterion<T>(
            _config.GetEffectiveMaxBudget());
        var poolCriterion = new StoppingCriteria.UnlabeledPoolExhaustedCriterion<T>();

        return StoppingCriteria.CompositeCriterion<T>.Any(budgetCriterion, poolCriterion);
    }

    private ActiveLearningResult<T> BuildFinalResult(
        List<ActiveLearningIterationResult<T>> iterationResults,
        int initialLabeled,
        string stoppingReason)
    {
        var learningCurve = GetLearningCurve();

        var lastCurvePoint = _learningCurvePoints.Count > 0 ? _learningCurvePoints[^1] : null;

        var result = new ActiveLearningResult<T>
        {
            TotalIterations = _iterationsCompleted,
            TotalSamplesLabeled = _labeledPool.Count,
            InitialLabeledSamples = initialLabeled,
            BudgetUsed = NumOps.FromDouble((double)_totalQueries / _config.GetEffectiveMaxBudget()),
            FinalTrainingAccuracy = lastCurvePoint != null ? lastCurvePoint.Accuracy : NumOps.Zero,
            FinalTrainingLoss = lastCurvePoint != null ? lastCurvePoint.Loss : NumOps.Zero,
            LearningCurve = learningCurve,
            IterationResults = iterationResults,
            StoppingReason = stoppingReason,
            TotalTime = _stopwatch.Elapsed,
            TotalTrainingTime = _totalTrainingTime,
            TotalSelectionTime = _totalSelectionTime,
            QueryStrategyName = _queryStrategy.Name,
            AreaUnderLearningCurve = learningCurve.AreaUnderCurve ?? NumOps.Zero
        };

        // Add validation metrics if available
        if (_validationSet != null && _validationSet.Count > 0 && lastCurvePoint != null)
        {
            result.FinalValidationAccuracy = lastCurvePoint.ValidationAccuracy;
            result.FinalValidationLoss = lastCurvePoint.ValidationLoss;
        }

        // Compute efficiency metrics
        result.SampleEfficiency = ComputeSampleEfficiency(learningCurve);

        return result;
    }

    private T ComputeAULC(LearningCurve<T> curve)
    {
        if (curve.SampleCounts.Length < 2)
        {
            return NumOps.Zero;
        }

        // Normalize sample counts to [0, 1] range
        int minSamples = curve.SampleCounts.Min();
        int maxSamples = curve.SampleCounts.Max();
        int range = maxSamples - minSamples;

        if (range == 0)
        {
            return curve.Accuracies.FirstOrDefault() ?? NumOps.Zero;
        }

        // Trapezoidal integration
        double auc = 0.0;
        for (int i = 1; i < curve.SampleCounts.Length; i++)
        {
            double x1 = (double)(curve.SampleCounts[i - 1] - minSamples) / range;
            double x2 = (double)(curve.SampleCounts[i] - minSamples) / range;
            double y1 = NumOps.ToDouble(curve.Accuracies[i - 1]);
            double y2 = NumOps.ToDouble(curve.Accuracies[i]);

            auc += (x2 - x1) * (y1 + y2) / 2.0;
        }

        return NumOps.FromDouble(auc);
    }

    private T ComputeSampleEfficiency(LearningCurve<T> curve)
    {
        if (curve.SampleCounts.Length < 2)
        {
            return NumOps.Zero;
        }

        // Sample efficiency = (final accuracy - initial accuracy) / samples added
        var initialAccuracy = NumOps.ToDouble(curve.Accuracies.First());
        var finalAccuracy = NumOps.ToDouble(curve.Accuracies.Last());
        var samplesAdded = curve.SampleCounts.Last() - curve.SampleCounts.First();

        if (samplesAdded <= 0)
        {
            return NumOps.Zero;
        }

        var efficiency = (finalAccuracy - initialAccuracy) / samplesAdded;
        return NumOps.FromDouble(efficiency);
    }

    private T ComputeSampleLoss(TOutput predicted, TOutput expected)
    {
        // Use the model's loss function for proper loss computation
        try
        {
            // Convert outputs to vectors for loss computation
            var predictedVector = ConversionsHelper.ConvertToVector<T, TOutput>(predicted);
            var expectedVector = ConversionsHelper.ConvertToVector<T, TOutput>(expected);

            // Use the model's default loss function for consistent loss computation
            return _model.DefaultLossFunction.CalculateLoss(predictedVector, expectedVector);
        }
        catch (InvalidOperationException)
        {
            // Fallback: For scalar types that can't be converted to vectors,
            // compute MSE directly
            if (predicted is T numPred && expected is T numExp)
            {
                var diff = NumOps.Subtract(numPred, numExp);
                return NumOps.Multiply(diff, diff);
            }
        }

        return NumOps.Zero;
    }

    protected virtual void OnIterationCompleted(ActiveLearningIterationResult<T> result)
    {
        IterationCompleted?.Invoke(this, result);
    }

    protected virtual void OnSamplesSelected(SamplesSelectedEventArgs<TInput> args)
    {
        SamplesSelected?.Invoke(this, args);
    }

    protected virtual void OnLearningCompleted(ActiveLearningResult<T> result)
    {
        LearningCompleted?.Invoke(this, result);
    }

    #endregion
}

/// <summary>
/// Internal class for tracking learning curve points.
/// </summary>
internal class LearningCurvePoint<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    /// <param name="numOps">Optional numeric operations provider. If null, a default provider will be used.</param>
    public LearningCurvePoint(INumericOperations<T>? numOps = null)
    {
        _numOps = numOps ?? MathHelper.GetNumericOperations<T>();
        SampleCount = 0;
        Accuracy = _numOps.Zero;
        Loss = _numOps.Zero;
        ValidationAccuracy = _numOps.Zero;
        ValidationLoss = _numOps.Zero;
        HasValidation = false;
    }

    public int SampleCount { get; set; }
    public T Accuracy { get; set; }
    public T Loss { get; set; }
    public T ValidationAccuracy { get; set; }
    public T ValidationLoss { get; set; }
    public bool HasValidation { get; set; }
}
