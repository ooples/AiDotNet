using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

internal sealed class TestMetaLearner : MetaLearnerBase<double, Matrix<double>, Vector<double>>
{
    public TestMetaLearner(
        IFullModel<double, Matrix<double>, Vector<double>> model,
        IMetaLearnerOptions<double> options,
        IEpisodicDataLoader<double, Matrix<double>, Vector<double>>? dataLoader = null,
        IGradientBasedOptimizer<double, Matrix<double>, Vector<double>>? metaOptimizer = null,
        IGradientBasedOptimizer<double, Matrix<double>, Vector<double>>? innerOptimizer = null)
        : base(model, new MeanSquaredErrorLoss<double>(), options, dataLoader, metaOptimizer, innerOptimizer)
    {
    }

    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MAML;

    public override double MetaTrain(TaskBatch<double, Matrix<double>, Vector<double>> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        var losses = new List<double>();
        foreach (var task in taskBatch.Tasks)
        {
            var predictions = MetaModel.Predict(task.SupportInput);
            var loss = ComputeLossFromOutput(predictions, task.SupportOutput);
            losses.Add(loss);

            var gradients = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            var updated = ApplyGradients(MetaModel.GetParameters(), gradients, _options.OuterLearningRate);
            MetaModel.SetParameters(updated);
        }

        return ComputeMean(losses);
    }

    public override IModel<Matrix<double>, Vector<double>, ModelMetadata<double>> Adapt(IMetaLearningTask<double, Matrix<double>, Vector<double>> task)
    {
        if (task == null)
            throw new ArgumentNullException(nameof(task));

        var adaptedModel = CloneModel();
        var parameters = adaptedModel.GetParameters();
        var gradients = ComputeGradients(adaptedModel, task.SupportInput, task.SupportOutput);
        parameters = ApplyGradients(parameters, gradients, _options.InnerLearningRate);
        adaptedModel.SetParameters(parameters);
        return adaptedModel;
    }

    public Vector<double> CallComputeGradients(IFullModel<double, Matrix<double>, Vector<double>> model, Matrix<double> input, Vector<double> output)
        => ComputeGradients(model, input, output);

    public Vector<double> CallComputeSecondOrderGradients(
        IFullModel<double, Matrix<double>, Vector<double>> model,
        List<(Matrix<double> input, Vector<double> target)> adaptationSteps,
        Matrix<double> queryInput,
        Vector<double> queryOutput,
        double innerLearningRate)
        => ComputeSecondOrderGradients(model, adaptationSteps, queryInput, queryOutput, innerLearningRate);

    public Vector<double> CallApplyGradients(Vector<double> parameters, Vector<double> gradients, double learningRate)
        => ApplyGradients(parameters, gradients, learningRate);

    public Vector<double> CallClipGradients(Vector<double> gradients, double? threshold = null)
        => ClipGradients(gradients, threshold);

    public double CallComputeAccuracy(Vector<double> predictions, Vector<double> labels)
        => ComputeAccuracy(predictions, labels);

    public double CallComputeLossFromOutput(Vector<double> predictions, Vector<double> expected)
        => ComputeLossFromOutput(predictions, expected);

    public Vector<double>? CallConvertToVector(Vector<double> output)
        => ConvertToVector(output);

    public TaskBatch<double, Matrix<double>, Vector<double>> CallCreateTaskBatch(IReadOnlyList<MetaLearningTask<double, Matrix<double>, Vector<double>>> tasks)
        => CreateTaskBatch(tasks);

    public IMetaLearningTask<double, Matrix<double>, Vector<double>> CallToMetaLearningTask(MetaLearningTask<double, Matrix<double>, Vector<double>> task)
        => ToMetaLearningTask(task);

    public IFullModel<double, Matrix<double>, Vector<double>> CallCloneModel()
        => CloneModel();

    public double CallComputeMean(List<double> values)
        => ComputeMean(values);
}

internal sealed class TestEpisodicDataLoader<T, TInput, TOutput> : IEpisodicDataLoader<T, TInput, TOutput>
{
    private readonly IReadOnlyList<MetaLearningTask<T, TInput, TOutput>> _tasks;
    private int _index;

    public TestEpisodicDataLoader(
        IReadOnlyList<MetaLearningTask<T, TInput, TOutput>> tasks,
        int nWay,
        int kShot,
        int queryShots,
        int availableClasses)
    {
        _tasks = tasks ?? throw new ArgumentNullException(nameof(tasks));
        NWay = nWay;
        KShot = kShot;
        QueryShots = queryShots;
        AvailableClasses = availableClasses;
        IsLoaded = true;
    }

    public string Name => "TestEpisodicDataLoader";

    public string Description => "In-memory episodic loader for integration tests.";

    public bool IsLoaded { get; private set; }

    public int BatchSize { get; set; } = 1;

    public bool HasNext => _tasks.Count > 0;

    public int TotalCount => _tasks.Count;

    public int CurrentIndex => Math.Min(_index, _tasks.Count);

    public int BatchCount => BatchSize <= 0 ? 0 : (int)Math.Ceiling((double)_tasks.Count / BatchSize);

    public int CurrentBatchIndex => BatchSize <= 0 ? 0 : _index / BatchSize;

    public double Progress => _tasks.Count == 0 ? 1.0 : (double)CurrentIndex / _tasks.Count;

    public int NWay { get; }

    public int KShot { get; }

    public int QueryShots { get; }

    public int AvailableClasses { get; }

    public Task LoadAsync(CancellationToken cancellationToken = default)
    {
        IsLoaded = true;
        return Task.CompletedTask;
    }

    public void Unload()
    {
        IsLoaded = false;
    }

    public void Reset()
    {
        _index = 0;
    }

    public MetaLearningTask<T, TInput, TOutput> GetNextTask()
    {
        if (_tasks.Count == 0)
            throw new InvalidOperationException("No tasks available.");

        var task = _tasks[_index % _tasks.Count];
        _index++;
        return task;
    }

    public IReadOnlyList<MetaLearningTask<T, TInput, TOutput>> GetTaskBatch(int numTasks)
    {
        if (numTasks <= 0)
            throw new ArgumentOutOfRangeException(nameof(numTasks));

        var batch = new List<MetaLearningTask<T, TInput, TOutput>>(numTasks);
        for (int i = 0; i < numTasks; i++)
        {
            batch.Add(GetNextTask());
        }
        return batch;
    }

    public void SetSeed(int seed)
    {
        _index = 0;
    }

    public MetaLearningTask<T, TInput, TOutput> GetNextBatch()
    {
        return GetNextTask();
    }

    public bool TryGetNextBatch(out MetaLearningTask<T, TInput, TOutput> batch)
    {
        if (_tasks.Count == 0)
        {
            batch = new MetaLearningTask<T, TInput, TOutput>();
            return false;
        }

        batch = GetNextTask();
        return true;
    }

    public IEnumerable<MetaLearningTask<T, TInput, TOutput>> GetBatches(int? batchSize = null, bool shuffle = true, bool dropLast = false, int? seed = null)
    {
        if (_tasks.Count == 0)
        {
            yield break;
        }

        foreach (var task in _tasks)
        {
            yield return task;
        }
    }

    public async IAsyncEnumerable<MetaLearningTask<T, TInput, TOutput>> GetBatchesAsync(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null,
        int prefetchCount = 2,
        CancellationToken cancellationToken = default)
    {
        if (_tasks.Count == 0)
        {
            yield break;
        }

        foreach (var task in _tasks)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return task;
            await Task.Yield();
        }
    }
}

internal sealed class ResetTrackingOptimizer : IGradientBasedOptimizer<double, Matrix<double>, Vector<double>>, IResettable
{
    public int ResetCount { get; private set; }

    public Vector<double> LastComputedGradients { get; private set; } = new Vector<double>(0);

    public OptimizationResult<double, Matrix<double>, Vector<double>> Optimize(OptimizationInputData<double, Matrix<double>, Vector<double>> inputData)
    {
        return new OptimizationResult<double, Matrix<double>, Vector<double>>();
    }

    public bool ShouldEarlyStop()
    {
        return false;
    }

    public OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>> GetOptions()
    {
        return new OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>();
    }

    public void Reset()
    {
        ResetCount++;
    }

    public Matrix<double> UpdateParameters(Matrix<double> parameters, Matrix<double> gradient)
    {
        return parameters;
    }

    public Vector<double> UpdateParameters(Vector<double> parameters, Vector<double> gradient)
    {
        return parameters;
    }

    public void UpdateParameters(List<ILayer<double>> layers)
    {
    }

    public IFullModel<double, Matrix<double>, Vector<double>> ApplyGradients(Vector<double> gradients, IFullModel<double, Matrix<double>, Vector<double>> model)
    {
        LastComputedGradients = gradients;
        return model;
    }

    public IFullModel<double, Matrix<double>, Vector<double>> ApplyGradients(
        Vector<double> originalParameters,
        Vector<double> gradients,
        IFullModel<double, Matrix<double>, Vector<double>> model)
    {
        LastComputedGradients = gradients;
        return model.WithParameters(originalParameters);
    }

    public Vector<double> ReverseUpdate(Vector<double> updatedParameters, Vector<double> appliedGradients)
    {
        return updatedParameters;
    }

    public byte[] Serialize()
    {
        return Array.Empty<byte>();
    }

    public void Deserialize(byte[] data)
    {
    }

    public void SaveModel(string filePath)
    {
        throw new NotSupportedException();
    }

    public void LoadModel(string filePath)
    {
        throw new NotSupportedException();
    }

    public bool SupportsGpuUpdate => false;

    public void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        throw new NotSupportedException("GPU operations are not supported in ResetTrackingOptimizer.");
    }

    public void InitializeGpuState(int parameterCount, IDirectGpuBackend backend)
    {
        throw new NotSupportedException("GPU operations are not supported in ResetTrackingOptimizer.");
    }

    public void DisposeGpuState()
    {
        // No GPU state to dispose
    }
}

/// <summary>
/// Test implementation of IMetaLearningTask for testing purposes.
/// </summary>
internal sealed class TestMetaLearningTask<T, TInput, TOutput> : IMetaLearningTask<T, TInput, TOutput>
{
    public TestMetaLearningTask(
        TInput supportInput,
        TOutput supportOutput,
        TInput queryInput,
        TOutput queryOutput,
        int numWays,
        int numShots,
        int numQueryPerClass,
        string? name = null)
    {
        SupportInput = supportInput;
        SupportOutput = supportOutput;
        QueryInput = queryInput;
        QueryOutput = queryOutput;
        NumWays = numWays;
        NumShots = numShots;
        NumQueryPerClass = numQueryPerClass;
        Name = name;
    }

    public TInput SupportInput { get; }
    public TOutput SupportOutput { get; }
    public TInput QueryInput { get; }
    public TOutput QueryOutput { get; }
    public string? Name { get; }
    public Dictionary<string, object>? Metadata { get; } = null;
    public int NumWays { get; }
    public int NumShots { get; }
    public int NumQueryPerClass { get; }
    public TInput QuerySetX => QueryInput;
    public TOutput QuerySetY => QueryOutput;
    public TInput SupportSetX => SupportInput;
    public TOutput SupportSetY => SupportOutput;
    public int? TaskId { get; set; }
}
