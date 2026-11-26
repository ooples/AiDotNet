using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.DistributedTraining;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <content>
/// Options the streaming path applies itself (LoRA, knowledge distillation, the post-training stages), and the
/// two places where a one-rank backend used to differ from no backend outside the optimizer step: the loader and
/// the lazy-layer warm-up.
/// </content>
public sealed partial class StreamingTrainingConfigurationTests
{
    /// <summary>
    /// A streaming loader that is not built on <c>StreamingDataLoaderBase</c> and yields batches of its own sizes,
    /// in dataset order, the way a length-bucketed or token-budget loader does.
    /// </summary>
    private sealed class OwnBatchSizesLoader : IStreamingDataLoader<float, Tensor<float>, Tensor<float>>
    {
        private readonly Tensor<float>[] _inputs;
        private readonly Tensor<float>[] _targets;
        private readonly int[] _batchSizes;

        public OwnBatchSizesLoader(Tensor<float>[] inputs, Tensor<float>[] targets, int[] batchSizes, int nominalBatchSize)
        {
            _inputs = inputs;
            _targets = targets;
            _batchSizes = batchSizes;
            BatchSize = nominalBatchSize;
        }

        public string Name => nameof(OwnBatchSizesLoader);

        public string Description => "Yields batches of fixed, uneven sizes in dataset order.";

        public bool IsLoaded { get; private set; }

        public int SampleCount => _inputs.Length;

        public int TotalCount => _inputs.Length;

        public int CurrentIndex => 0;

        public int BatchCount => _batchSizes.Length;

        public int CurrentBatchIndex => 0;

        public double Progress => 0.0;

        public int BatchSize { get; set; }

        public int PrefetchCount => 1;

        public int NumWorkers => 1;

        public Task LoadAsync(CancellationToken cancellationToken = default)
        {
            IsLoaded = true;
            return Task.CompletedTask;
        }

        public void Unload() => IsLoaded = false;

        public void Reset()
        {
        }

        public IEnumerable<(Tensor<float>[] Inputs, Tensor<float>[] Outputs)> GetBatches(
            bool shuffle = true,
            bool dropLast = false,
            int? seed = null)
        {
            int offset = 0;
            foreach (int size in _batchSizes)
            {
                int take = Math.Min(size, _inputs.Length - offset);
                if (take <= 0)
                {
                    yield break;
                }

                yield return (_inputs.Skip(offset).Take(take).ToArray(), _targets.Skip(offset).Take(take).ToArray());
                offset += take;
            }
        }

        public async IAsyncEnumerable<(Tensor<float>[] Inputs, Tensor<float>[] Outputs)> GetBatchesAsync(
            bool shuffle = true,
            bool dropLast = false,
            int? seed = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (var batch in GetBatches(shuffle, dropLast, seed))
            {
                cancellationToken.ThrowIfCancellationRequested();
                await Task.Yield();
                yield return batch;
            }
        }
    }

    [Fact(Timeout = 90000)]
    public async Task NeuralNetwork_OneRankBackend_ReadsALoaderWithItsOwnBatchSizesAsNoBackendDoes()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 20, seed: 401);
        const long steps = 8;

        async Task<Vector<float>> TrainAsync(int[] batchSizes, bool oneRankBackend)
        {
            var optimizer = CreateAdam();
            var model = CreateMlp(optimizer);
            model.SetParameters(CreateInitialWeights(model.GetParameters().Length, seed: 402));
            IAiModelBuilder<float, Tensor<float>, Tensor<float>> builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureDataLoader(new OwnBatchSizesLoader(inputs, targets, batchSizes, nominalBatchSize: 6))
                .ConfigureGpuAcceleration(CpuOnlyConfig())
                .ConfigureStreamingTraining(StepOptions(steps));
            if (!oneRankBackend)
            {
                await builder.BuildAsync();
                return model.GetParameters();
            }

            var backend = new InMemoryCommunicationBackend<float>(rank: 0, worldSize: 1, "own_batches_" + Guid.NewGuid().ToString("N"));
            try
            {
                await builder.ConfigureDistributedTraining(backend, DistributedStrategy.DDP).BuildAsync();
                return model.GetParameters();
            }
            finally
            {
                Shutdown(new[] { backend });
            }
        }

        int[] ownSizes = { 5, 3, 7, 5 };
        Vector<float> expected = await TrainAsync(ownSizes, oneRankBackend: false);
        Vector<float> actual = await TrainAsync(ownSizes, oneRankBackend: true);
        double diff = MaxAbsDiff(expected, actual);
        Assert.True(diff == 0.0, $"A one-rank backend changed the result of a loader with its own batch sizes by {diff:R}.");

        // Re-cut to the nominal batch size of 6, the same samples train to different weights, so the equality
        // above would not hold if the one-rank run had re-chunked the loader.
        Vector<float> rechunked = await TrainAsync(new[] { 6, 6, 6, 2 }, oneRankBackend: false);
        Assert.True(MaxAbsDiff(expected, rechunked) > 1e-5,
            "This data cannot tell the loader's own batches from batches re-cut to the nominal size.");
    }

    /// <summary>Keeps the first half of the features, so the model trains on a narrower input than the loader yields.</summary>
    private sealed class FirstHalfOfFeatures : IDataTransformer<float, Tensor<float>, Tensor<float>>
    {
        public bool IsFitted { get; private set; }

        public int[]? ColumnIndices => null;

        public bool SupportsInverseTransform => false;

        public void Fit(Tensor<float> data) => IsFitted = true;

        public Tensor<float> Transform(Tensor<float> data)
        {
            int width = data.Shape[data.Shape.Length - 1];
            int kept = width / 2;
            if (data.Shape.Length == 1)
            {
                var sample = new Tensor<float>(new[] { kept });
                for (int d = 0; d < kept; d++)
                {
                    sample[d] = data[d];
                }

                return sample;
            }

            int rows = data.Shape[0];
            var batch = new Tensor<float>(new[] { rows, kept });
            for (int r = 0; r < rows; r++)
            {
                for (int d = 0; d < kept; d++)
                {
                    batch[r, d] = data[r, d];
                }
            }

            return batch;
        }

        public Tensor<float> FitTransform(Tensor<float> data)
        {
            Fit(data);
            return Transform(data);
        }

        public Tensor<float> InverseTransform(Tensor<float> data) =>
            throw new NotSupportedException("The dropped features cannot be restored.");

        public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) =>
            (inputFeatureNames ?? Array.Empty<string>()).Take((inputFeatureNames?.Length ?? 0) / 2).ToArray();
    }

    [Theory(Timeout = 90000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task LazyLayers_BehindAnUnfittedPipeline_AreSizedForTheTransformedInput(bool oneRankBackend)
    {
        var (inputs, targets) = CreateDataset(sampleCount: 16, seed: 411);
        const int transformedWidth = InDim / 2;

        var optimizer = CreateAdam();
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(outputSize: HiddenDim, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(outputSize: OutDim, activationFunction: new IdentityActivation<float>()),
        };
        var model = new FeedForwardNeuralNetwork<float>(
            new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: transformedWidth,
                outputSize: OutDim,
                layers: layers),
            optimizer: optimizer,
            lossFunction: new MeanSquaredErrorLoss<float>());
        Assert.True(model.HasLazyUninitializedLayers(), "The layers must still be unsized, or there is no warm-up to test.");

        IAiModelBuilder<float, Tensor<float>, Tensor<float>> builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 4))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigurePreprocessing(new FirstHalfOfFeatures())
            .ConfigureStreamingTraining(StepOptions(4));

        var backend = new InMemoryCommunicationBackend<float>(rank: 0, worldSize: 1, "lazy_pipeline_" + Guid.NewGuid().ToString("N"));
        try
        {
            if (oneRankBackend)
            {
                builder = builder.ConfigureDistributedTraining(backend, DistributedStrategy.DDP);
            }

            await builder.BuildAsync();
        }
        finally
        {
            Shutdown(new[] { backend });
        }

        // Sized on the raw sample the first layer would hold InDim * HiddenDim weights, and the first transformed
        // batch would not fit it.
        int expectedParameters = (transformedWidth * HiddenDim + HiddenDim) + (HiddenDim * OutDim + OutDim);
        Assert.Equal(expectedParameters, model.GetParameters().Length);
        Assert.False(float.IsNaN(model.GetLastLoss()), "Training on the transformed input produced no loss.");
    }

    [Theory(Timeout = 120000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task LoRA_OnStreaming_TrainsTheAdaptersAndLeavesTheBaseWeightsFrozen(bool lazyLayers)
    {
        var (inputs, targets) = CreateDataset(sampleCount: 24, seed: 421);
        var optimizer = CreateAdam();
        FeedForwardNeuralNetwork<float> model;
        if (lazyLayers)
        {
            // Nothing has sized the dense layers, so the adapters can only be built after the warm-up forward on a
            // sample peeked from the stream.
            model = new FeedForwardNeuralNetwork<float>(
                new NeuralNetworkArchitecture<float>(
                    inputType: InputType.OneDimensional,
                    taskType: NeuralNetworkTaskType.Regression,
                    inputSize: InDim,
                    outputSize: OutDim,
                    layers: new List<ILayer<float>>
                    {
                        new DenseLayer<float>(outputSize: HiddenDim, activationFunction: new ReLUActivation<float>()),
                        new DenseLayer<float>(outputSize: OutDim, activationFunction: new IdentityActivation<float>()),
                    }),
                optimizer: optimizer,
                lossFunction: new MeanSquaredErrorLoss<float>());
        }
        else
        {
            model = CreateMlp(optimizer);
            model.SetParameters(CreateInitialWeights(model.GetParameters().Length, seed: 422));
        }

        ILayer<float>[] denseLayers = model.Layers.Where(layer => layer is DenseLayer<float>).ToArray();
        Assert.Equal(2, denseLayers.Length);
        Vector<float>[] baseBefore = lazyLayers
            ? Array.Empty<Vector<float>>()
            : denseLayers.Select(layer => layer.GetParameters().Clone()).ToArray();
        var probeInput = new Tensor<float>(new[] { 1, InDim });
        for (int d = 0; d < InDim; d++)
        {
            probeInput[0, d] = 0.1f * (d + 1);
        }

        Vector<float>? predictionBefore = lazyLayers ? null : model.Predict(probeInput).ToVector();

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 8))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureLoRA(new DefaultLoRAConfiguration<float>(rank: 2, alpha: 2, freezeBaseLayer: true))
            .ConfigureStreamingTraining(StepOptions(9))
            .BuildAsync();

        // Both dense layers were swapped for adapters around the original layers.
        LoRAAdapterBase<float>[] adapters = model.Layers.OfType<LoRAAdapterBase<float>>().ToArray();
        Assert.Equal(2, adapters.Length);
        Assert.Empty(model.Layers.Where(layer => layer is DenseLayer<float>));
        for (int i = 0; i < adapters.Length; i++)
        {
            Assert.True(adapters[i].IsBaseLayerFrozen);
            Assert.Same(denseLayers[i], adapters[i].BaseLayer);
        }

        // The adapters trained: a fresh adapter adds nothing to its base layer (its B matrix starts at zero), so a
        // B matrix away from zero is the mark of training.
        foreach (var adapter in adapters)
        {
            Matrix<float> matrixB = adapter.LoRALayer.GetMatrixB();
            Assert.True(
                Enumerable.Range(0, matrixB.Rows).Any(r => Enumerable.Range(0, matrixB.Columns).Any(c => matrixB[r, c] != 0f)),
                "The adapter's B matrix is still zero, so the adapter was never trained.");
        }

        if (lazyLayers)
        {
            return;
        }

        // The base weights are bit for bit what they were, and the output moved, so the adapters did the moving.
        for (int i = 0; i < adapters.Length; i++)
        {
            double baseDrift = MaxAbsDiff(baseBefore[i], adapters[i].BaseLayer.GetParameters());
            Assert.True(baseDrift == 0.0, $"Base layer {i} changed by {baseDrift:R} although LoRA froze it.");
        }

        Vector<float> before = predictionBefore ?? throw new InvalidOperationException("The prediction before training was not recorded.");
        Assert.True(MaxAbsDiff(before, model.Predict(probeInput).ToVector()) > 1e-4,
            "The model's output did not change, so nothing trained.");
    }

    [Fact(Timeout = 120000)]
    public async Task KnowledgeDistillation_OnStreaming_MovesTheStudentTowardTheTeacher()
    {
        const int cols = 4, outs = 2, rows = 40;
        var inputs = new Tensor<double>[rows];
        var targets = new Tensor<double>[rows];
        for (int i = 0; i < rows; i++)
        {
            inputs[i] = new Tensor<double>(new[] { cols });
            targets[i] = new Tensor<double>(new[] { outs });
            for (int j = 0; j < cols; j++) inputs[i][j] = Math.Sin((i + j) * 0.2);
            for (int o = 0; o < outs; o++) targets[i][o] = Math.Cos((i + o) * 0.2);
        }

        // A fixed teacher distinct from the labels, and Alpha = 0 (teacher only), so a student that trained on the
        // labels instead would not reduce this loss.
        Tensor<double> Teacher(Tensor<double> input)
        {
            var logits = new Tensor<double>(new[] { 1, outs });
            logits[0, 0] = 2.0;
            logits[0, 1] = -1.0;
            return logits;
        }

        var student = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: cols, outputSize: outs,
            layers: new List<ILayer<double>>
            {
                new DenseLayer<double>(6, (IActivationFunction<double>)new ReLUActivation<double>()),
                new DenseLayer<double>(outs),
            }));
        var strategy = DistillationStrategyFactory<double>.CreateResponseBasedStrategy(2.0, 0.0);

        double DistillationLoss()
        {
            double total = 0.0;
            for (int i = 0; i < rows; i++)
            {
                var row = new Tensor<double>(new[] { 1, cols });
                for (int j = 0; j < cols; j++) row[0, j] = inputs[i][j];
                var prediction = student.Predict(row);
                var teacher = Teacher(row);
                var studentRow = new Matrix<double>(1, outs);
                var teacherRow = new Matrix<double>(1, outs);
                for (int k = 0; k < outs; k++)
                {
                    studentRow[0, k] = prediction[0, k];
                    teacherRow[0, k] = teacher[0, k];
                }

                total += strategy.ComputeLoss(studentRow, teacherRow, null);
            }

            return total / rows;
        }

        double before = DistillationLoss();
        await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(student)
            .ConfigureOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
                null,
                new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.05, MaxIterations = 10 }))
            .ConfigureKnowledgeDistillation(new KnowledgeDistillationOptions<double, Tensor<double>, Tensor<double>>
            {
                TeacherForward = Teacher,
                Temperature = 2.0,
                Alpha = 0.0,
            })
            .ConfigureDataLoader(new StreamingDataLoader<double, Tensor<double>, Tensor<double>>(
                sampleCount: rows,
                sampleReader: (idx, ct) => Task.FromResult((inputs[idx], targets[idx])),
                batchSize: 8,
                prefetchCount: 1,
                numWorkers: 1))
            .BuildAsync();
        double after = DistillationLoss();

        Assert.True(after < before * 0.5,
            $"Distillation on the stream did not bring the student toward the teacher (loss {before:R} before, {after:R} after).");
    }

    [Fact(Timeout = 90000)]
    public async Task KnowledgeDistillation_OnStreaming_WithoutATeacher_FailsClearly()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 8, seed: 431);
        var optimizer = CreateAdam();
        var model = CreateMlp(optimizer);

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 4))
                .ConfigureGpuAcceleration(CpuOnlyConfig())
                .ConfigureDistillationStrategy(DistillationStrategyFactory<float>.CreateResponseBasedStrategy(2.0, 0.0))
                .ConfigureStreamingTraining(StepOptions(2))
                .BuildAsync());
        Assert.Contains("teacher", refused.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 90000)]
    public async Task TrainingPipeline_OnStreaming_RunsItsStagesOnTheStreamTrainedModel()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 16, seed: 441);
        var optimizer = CreateAdam();
        var model = CreateMlp(optimizer);
        Vector<float> initial = CreateInitialWeights(model.GetParameters().Length, seed: 442);
        model.SetParameters(initial.Clone());

        var stageData = new FineTuningData<float, Tensor<float>, Tensor<float>>
        {
            Inputs = new[] { inputs[0] },
            Outputs = new[] { targets[0] },
        };
        int stageCalls = 0;
        Vector<float>? weightsSeenByStage = null;
        var stage = new TrainingStage<float, Tensor<float>, Tensor<float>>
        {
            Name = "after-the-stream",
            Enabled = true,
            TrainingData = stageData,
            CustomTrainingFunction = (stageModel, data, ct) =>
            {
                stageCalls++;
                Assert.Same(stageData, data);
                weightsSeenByStage = stageModel.GetParameters().Clone();

                // The stage's own change to the model: every weight halved.
                stageModel.SetParameters(stageModel.GetParameters().Multiply(0.5f));
                return Task.FromResult(stageModel);
            },
        };

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 4))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureTrainingPipeline(new TrainingPipelineConfiguration<float, Tensor<float>, Tensor<float>>
            {
                Stages = new List<TrainingStage<float, Tensor<float>, Tensor<float>>> { stage },
            })
            .ConfigureStreamingTraining(StepOptions(5))
            .BuildAsync();

        Assert.Equal(1, stageCalls);
        Vector<float> seen = weightsSeenByStage ?? throw new InvalidOperationException("The stage did not record the weights it was given.");
        Assert.True(MaxAbsDiff(initial, seen) > 1e-3, "The stage ran on a model the stream had not trained yet.");
        Assert.True(MaxAbsDiff(seen.Multiply(0.5f), model.GetParameters()) == 0.0,
            "The built model does not carry the stage's change.");
    }

    [Fact(Timeout = 90000)]
    public async Task NeuralNetwork_TwoRanks_MatchOneProcessThatAccumulatesTheSameShards()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 24, seed: 461);
        const long steps = 6;
        Vector<float> Initial(int length) => CreateInitialWeights(length, seed: 462);

        // One process, two micro-batches of 4 per update: the same two shards, in the same order, that two ranks
        // of 4 samples each reduce. The two are one rounding apart, not bit for bit: the ranks multiply each shard's
        // gradient by its share of the samples before the sum, the accumulating process sums first and divides
        // once. That is one unit in the last place of a float per step (3e-8 measured over these six steps).
        var referenceOptimizer = CreateAdam();
        var reference = CreateMlp(referenceOptimizer);
        reference.SetParameters(Initial(reference.GetParameters().Length));
        var referenceOptions = StepOptions(steps);
        referenceOptions.GradientAccumulationSteps = 2;
        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(reference)
            .ConfigureOptimizer(referenceOptimizer)
            .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 4))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureStreamingTraining(referenceOptions)
            .BuildAsync();

        string session = "shards_" + Guid.NewGuid().ToString("N");
        var backends = new ICommunicationBackend<float>[]
        {
            new InMemoryCommunicationBackend<float>(0, 2, session),
            new InMemoryCommunicationBackend<float>(1, 2, session),
        };
        try
        {
            var models = new FeedForwardNeuralNetwork<float>[2];
            await Task.WhenAll(Enumerable.Range(0, 2).Select(rank => Task.Run(async () =>
            {
                var optimizer = CreateAdam();
                var model = CreateMlp(optimizer);
                model.SetParameters(Initial(model.GetParameters().Length));
                models[rank] = model;
                await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                    .ConfigureModel(model)
                    .ConfigureOptimizer(optimizer)
                    .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 4))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureDistributedTraining(backends[rank], DistributedStrategy.DDP)
                    .ConfigureStreamingTraining(StepOptions(steps))
                    .BuildAsync();
            })));

            Assert.True(MaxAbsDiff(models[0].GetParameters(), models[1].GetParameters()) == 0.0, "The ranks diverged.");
            double moved = MaxAbsDiff(Initial(reference.GetParameters().Length), reference.GetParameters());
            Assert.True(moved > 1e-3, $"The one-process reference barely trained (largest weight change {moved:R}).");
            double diff = MaxAbsDiff(reference.GetParameters(), models[0].GetParameters());
            Assert.True(diff < 1e-6,
                $"Two ranks differ from one process accumulating the same two shards by {diff:R}.");
        }
        finally
        {
            Shutdown(backends);
        }
    }

    [Fact(Timeout = 90000)]
    public async Task TrainersWithNoGradientExchange_OnSeveralRanks_AreRefusedByName()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 16, seed: 451);
        string session = "unsynchronized_" + Guid.NewGuid().ToString("N");
        var backends = new ICommunicationBackend<float>[]
        {
            new InMemoryCommunicationBackend<float>(0, 2, session),
            new InMemoryCommunicationBackend<float>(1, 2, session),
        };
        try
        {
            var refusals = await Task.WhenAll(Enumerable.Range(0, 2).Select(rank => Task.Run(async () =>
            {
                var optimizer = CreateAdam();
                var model = CreateMlp(optimizer);
                Vector<float> before = model.GetParameters();
                var refused = await Assert.ThrowsAsync<NotSupportedException>(() =>
                    new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                        .ConfigureModel(model)
                        .ConfigureOptimizer(optimizer)
                        .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 4))
                        .ConfigureGpuAcceleration(CpuOnlyConfig())
                        .ConfigureDistributedTraining(backends[rank], DistributedStrategy.DDP)
                        .ConfigureKnowledgeDistillation(new KnowledgeDistillationOptions<float, Tensor<float>, Tensor<float>>
                        {
                            TeacherForward = input => input,
                        })
                        .ConfigureTrainingPipeline(new TrainingPipelineConfiguration<float, Tensor<float>, Tensor<float>>
                        {
                            Stages = new List<TrainingStage<float, Tensor<float>, Tensor<float>>>
                            {
                                new TrainingStage<float, Tensor<float>, Tensor<float>> { Name = "stage", Enabled = true },
                            },
                        })
                        .ConfigureStreamingTraining(StepOptions(2))
                        .BuildAsync());
                Assert.True(MaxAbsDiff(before, model.GetParameters()) == 0.0, "The refused build still trained the model.");
                return refused.Message;
            })));

            foreach (string message in refusals)
            {
                Assert.Contains(nameof(IAiModelBuilder<float, Tensor<float>, Tensor<float>>.ConfigureKnowledgeDistillation), message, StringComparison.Ordinal);
                Assert.Contains(nameof(IAiModelBuilder<float, Tensor<float>, Tensor<float>>.ConfigureTrainingPipeline), message, StringComparison.Ordinal);
                Assert.Contains("2 ranks", message, StringComparison.Ordinal);
            }
        }
        finally
        {
            Shutdown(backends);
        }
    }
}
