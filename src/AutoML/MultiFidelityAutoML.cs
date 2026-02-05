using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.AutoML;

/// <summary>
/// Built-in AutoML strategy that uses multi-fidelity (successive halving) and ASHA scheduling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This strategy evaluates many candidate configurations on a reduced training budget first (for example, a smaller
/// subset of rows), then promotes only the most promising trials to higher budgets.
/// </para>
/// <para>
/// <b>ASHA (Asynchronous Successive Halving Algorithm)</b> extends this with:
/// <list type="bullet">
/// <item><description>Parallel trial execution at each fidelity rung.</description></item>
/// <item><description>Per-trial early stopping of underperforming configurations.</description></item>
/// <item><description>Grace periods to allow trials to "warm up" before stopping.</description></item>
/// </list>
/// Enable ASHA via <see cref="AutoMLMultiFidelityOptions.EnableAsyncExecution"/>.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is a "try cheap first, then spend more on the best" strategy:
/// <list type="number">
/// <item><description>Try many models quickly (small subset).</description></item>
/// <item><description>Keep the best few.</description></item>
/// <item><description>Re-train those on more data.</description></item>
/// <item><description>Repeat until full training data.</description></item>
/// </list>
/// </para>
/// </remarks>
public sealed class MultiFidelityAutoML<T, TInput, TOutput> : BuiltInSupervisedAutoMLModelBase<T, TInput, TOutput>
{
    private const string FidelityFractionKey = "FidelityFraction";

    private readonly AutoMLMultiFidelityOptions _options;

    public MultiFidelityAutoML(
        Random? random = null,
        AutoMLMultiFidelityOptions? options = null)
        : base(random)
    {
        _options = options ?? new AutoMLMultiFidelityOptions();
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

            var fidelityFractions = ResolveFidelityFractions(_options);
            if (TrialLimit < fidelityFractions.Length)
            {
                fidelityFractions = fidelityFractions
                    .Skip(fidelityFractions.Length - TrialLimit)
                    .ToArray();
            }

            double reductionFactor = _options.ReductionFactor;
            if (double.IsNaN(reductionFactor) || double.IsInfinity(reductionFactor) || reductionFactor < 2.0)
            {
                reductionFactor = 2.0;
            }

            var rungCounts = AllocateRungTrialCounts(TrialLimit, fidelityFractions.Length, reductionFactor);

            // Precompute a single training permutation so data subsets are nested across fidelities.
            int sampleCount = GetSampleCount(inputs);
            var shuffledTrainingRowIndices = CreateShuffledIndices(sampleCount);

            var trialConfigs = new List<Dictionary<string, object>>();
            for (int i = 0; i < rungCounts[0]; i++)
            {
                trialConfigs.Add(await SuggestNextTrialAsync());
            }

            IFullModel<T, TInput, TOutput>? bestFullFidelityModel = null;
            double bestFullFidelityScore = _maximize ? double.NegativeInfinity : double.PositiveInfinity;

            // Resolve ASHA parallelism settings
            int maxParallelism = _options.EnableAsyncExecution
                ? (_options.MaxParallelism > 0 ? _options.MaxParallelism : Environment.ProcessorCount)
                : 1;

            for (int rungIndex = 0; rungIndex < fidelityFractions.Length; rungIndex++)
            {
                if (DateTime.UtcNow >= deadline)
                {
                    break;
                }

                cancellationToken.ThrowIfCancellationRequested();

                double fraction = fidelityFractions[rungIndex];
                int subsetSize = ResolveSubsetSize(sampleCount, fraction);

                var rungResults = new List<(Dictionary<string, object> Config, double Score, bool Success)>();
                var rungResultsLock = new object();

                if (_options.EnableAsyncExecution && maxParallelism > 1)
                {
                    // ASHA-style async parallel execution
                    var parallelResult = await ExecuteTrialsInParallelAsync(
                        trialConfigs,
                        inputs,
                        targets,
                        validationInputs,
                        validationTargets,
                        fraction,
                        subsetSize,
                        shuffledTrainingRowIndices,
                        rungResults,
                        rungResultsLock,
                        maxParallelism,
                        deadline,
                        bestFullFidelityModel,
                        bestFullFidelityScore,
                        cancellationToken);

                    bestFullFidelityModel = parallelResult.BestModel;
                    bestFullFidelityScore = parallelResult.BestScore;
                }
                else
                {
                    // Sequential execution (original behavior)
                    foreach (var config in trialConfigs)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        if (DateTime.UtcNow >= deadline)
                        {
                            break;
                        }

                        var (score, success) = await ExecuteSingleTrialAsync(
                            config,
                            inputs,
                            targets,
                            validationInputs,
                            validationTargets,
                            fraction,
                            subsetSize,
                            shuffledTrainingRowIndices,
                            cancellationToken);

                        rungResults.Add((config, score, success));

                        if (fraction >= 1.0 - 1e-12 && success)
                        {
                            bool improved = _maximize ? score > bestFullFidelityScore : score < bestFullFidelityScore;
                            if (improved)
                            {
                                bestFullFidelityScore = score;
                                bestFullFidelityModel = BestModel;
                            }
                        }
                    }
                }

                if (rungIndex == fidelityFractions.Length - 1)
                {
                    break;
                }

                int nextCount = rungCounts[rungIndex + 1];
                var promotable = rungResults
                    .Where(r => r.Success)
                    .OrderByDescending(r => _maximize ? r.Score : -r.Score)
                    .Take(Math.Min(nextCount, rungResults.Count))
                    .Select(r => r.Config)
                    .ToList();

                if (promotable.Count == 0)
                {
                    break;
                }

                trialConfigs = promotable;
            }

            if (bestFullFidelityModel is not null)
            {
                BestModel = bestFullFidelityModel;
                BestScore = bestFullFidelityScore;
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

    /// <summary>
    /// Executes a single trial and returns the score and success status.
    /// </summary>
    private async Task<(double Score, bool Success)> ExecuteSingleTrialAsync(
        Dictionary<string, object> config,
        TInput inputs,
        TOutput targets,
        TInput validationInputs,
        TOutput validationTargets,
        double fraction,
        int subsetSize,
        int[] shuffledTrainingRowIndices,
        CancellationToken cancellationToken)
    {
        if (!config.TryGetValue("ModelType", out var modelTypeObj) || modelTypeObj is not ModelType modelType)
        {
            throw new InvalidOperationException("AutoML trial parameters must include a ModelType entry.");
        }

        var (rungTrainInputs, rungTrainTargets) = CreateRungTrainingSubset(
            inputs,
            targets,
            modelType,
            subsetSize,
            shuffledTrainingRowIndices);

        var trialParameters = new Dictionary<string, object>(config, StringComparer.Ordinal)
        {
            [FidelityFractionKey] = fraction
        };

        var score = await ExecuteTrialAsync(modelType, trialParameters, rungTrainInputs, rungTrainTargets, validationInputs, validationTargets, cancellationToken);

        bool success;
        lock (_lock)
        {
            success = _trialHistory.LastOrDefault()?.Success ?? false;
        }

        return (score, success);
    }

    /// <summary>
    /// Executes trials in parallel using ASHA-style async execution.
    /// </summary>
    /// <returns>A tuple containing the best model and score found during parallel execution.</returns>
    private async Task<(IFullModel<T, TInput, TOutput>? BestModel, double BestScore)> ExecuteTrialsInParallelAsync(
        List<Dictionary<string, object>> trialConfigs,
        TInput inputs,
        TOutput targets,
        TInput validationInputs,
        TOutput validationTargets,
        double fraction,
        int subsetSize,
        int[] shuffledTrainingRowIndices,
        List<(Dictionary<string, object> Config, double Score, bool Success)> rungResults,
        object rungResultsLock,
        int maxParallelism,
        DateTime deadline,
        IFullModel<T, TInput, TOutput>? currentBestModel,
        double currentBestScore,
        CancellationToken cancellationToken)
    {
        using var semaphore = new SemaphoreSlim(maxParallelism, maxParallelism);
        var tasks = new List<Task>();

        // Track best model in a thread-safe way
        var localBestModel = currentBestModel;
        var localBestScore = currentBestScore;
        var bestLock = new object();

        foreach (var config in trialConfigs)
        {
            if (DateTime.UtcNow >= deadline)
            {
                break;
            }

            cancellationToken.ThrowIfCancellationRequested();

            await semaphore.WaitAsync(cancellationToken);

            var task = Task.Run(async () =>
            {
                try
                {
                    if (DateTime.UtcNow >= deadline)
                    {
                        return;
                    }

                    var (score, success) = await ExecuteSingleTrialAsync(
                        config,
                        inputs,
                        targets,
                        validationInputs,
                        validationTargets,
                        fraction,
                        subsetSize,
                        shuffledTrainingRowIndices,
                        cancellationToken);

                    lock (rungResultsLock)
                    {
                        rungResults.Add((config, score, success));
                    }

                    if (fraction >= 1.0 - 1e-12 && success)
                    {
                        lock (bestLock)
                        {
                            bool improved = _maximize ? score > localBestScore : score < localBestScore;
                            if (improved)
                            {
                                localBestScore = score;
                                localBestModel = BestModel;
                            }
                        }
                    }
                }
                finally
                {
                    semaphore.Release();
                }
            }, cancellationToken);

            tasks.Add(task);
        }

        await Task.WhenAll(tasks);

        // Return the best model and score found
        return (localBestModel, localBestScore);
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
        return new MultiFidelityAutoML<T, TInput, TOutput>(Random, _options);
    }

    private static double[] ResolveFidelityFractions(AutoMLMultiFidelityOptions options)
    {
        if (options is null)
        {
            return new[] { 0.25, 0.5, 1.0 };
        }

        if (options.TrainingFractions is null || options.TrainingFractions.Length == 0)
        {
            return new[] { 0.25, 0.5, 1.0 };
        }

        var distinct = options.TrainingFractions
            .Select(f => NormalizeFraction(f))
            .Distinct()
            .OrderBy(f => f)
            .ToList();

        if (distinct.Count == 0)
        {
            return new[] { 0.25, 0.5, 1.0 };
        }

        if (distinct[distinct.Count - 1] < 1.0)
        {
            distinct.Add(1.0);
        }
        else
        {
            distinct[distinct.Count - 1] = 1.0;
        }

        return distinct.ToArray();
    }

    private static double NormalizeFraction(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(value), "Training fraction must be in (0, 1].");
        }

        return value > 1.0 ? 1.0 : value;
    }

    private static int[] AllocateRungTrialCounts(int totalTrials, int rungCount, double reductionFactor)
    {
        if (rungCount <= 0)
        {
            return Array.Empty<int>();
        }

        if (totalTrials <= 0)
        {
            return new int[rungCount];
        }

        if (rungCount <= 1)
        {
            return new[] { totalTrials };
        }

        double weightSum = 0.0;
        for (int i = 0; i < rungCount; i++)
        {
            weightSum += 1.0 / Math.Pow(reductionFactor, i);
        }

        int baseCount = (int)Math.Floor(totalTrials / Math.Max(1e-9, weightSum));
        baseCount = Math.Max(1, baseCount);

        var counts = new int[rungCount];
        int allocated = 0;

        for (int i = 0; i < rungCount; i++)
        {
            int raw = (int)Math.Floor(baseCount / Math.Pow(reductionFactor, i));
            counts[i] = i == 0 ? Math.Max(1, raw) : Math.Max(0, raw);
            allocated += counts[i];
        }

        // If rounding pushed us over budget, decrement from later rungs first, keeping rung 0 at >= 1.
        for (int i = rungCount - 1; allocated > totalTrials && i >= 0; i--)
        {
            int min = (i == 0) ? 1 : 0;
            while (allocated > totalTrials && counts[i] > min)
            {
                counts[i]--;
                allocated--;
            }
        }

        // If rounding left us under budget, add remaining trials starting from rung 0.
        // Early rungs are cheaper (lower fidelity), so they should get extra exploration budget.
        while (allocated < totalTrials)
        {
            for (int i = 0; i < rungCount && allocated < totalTrials; i++)
            {
                counts[i]++;
                allocated++;
            }
        }

        return counts;
    }

    private static int ResolveSubsetSize(int sampleCount, double fraction)
    {
        if (fraction >= 1.0)
        {
            return sampleCount;
        }

        int subset = (int)Math.Round(sampleCount * fraction);
        return Math.Max(1, Math.Min(sampleCount, subset));
    }

    private static int GetSampleCount(TInput inputs)
    {
        if (inputs is null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (inputs is not Matrix<T> matrix)
        {
            throw new NotSupportedException(
                $"Multi-fidelity built-in AutoML currently supports Matrix<T>/Vector<T> tasks. Received {typeof(TInput).Name}/{typeof(TOutput).Name}.");
        }

        return matrix.Rows;
    }

    private int[] CreateShuffledIndices(int sampleCount)
    {
        var indices = Enumerable.Range(0, sampleCount).ToArray();

        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = Random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        return indices;
    }

    private static (TInput Inputs, TOutput Targets) CreateRungTrainingSubset(
        TInput inputs,
        TOutput targets,
        ModelType modelType,
        int subsetSize,
        int[] shuffledRowIndices)
    {
        if (inputs is null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (targets is null)
        {
            throw new ArgumentNullException(nameof(targets));
        }

        if (inputs is not Matrix<T> x || targets is not Vector<T> y)
        {
            throw new NotSupportedException(
                $"Multi-fidelity built-in AutoML currently supports Matrix<T>/Vector<T> tasks. Received {typeof(TInput).Name}/{typeof(TOutput).Name}.");
        }

        if (subsetSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(subsetSize));
        }

        if (subsetSize >= x.Rows)
        {
            return (inputs, targets);
        }

        IEnumerable<int> indices = IsTimeSeriesModel(modelType)
            ? Enumerable.Range(0, subsetSize)
            : shuffledRowIndices.Take(subsetSize);

        var xs = x.GetRows(indices);
        var ys = y.GetElements(indices);

        return ((TInput)(object)xs, (TOutput)(object)ys);
    }

    private static bool IsTimeSeriesModel(ModelType modelType)
    {
        return modelType == ModelType.TimeSeriesRegression || modelType == ModelType.BayesianStructuralTimeSeriesModel;
    }
}
