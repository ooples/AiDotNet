using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.CrossValidators;
using AiDotNet.Data.Loaders;
using AiDotNet.DistributedTraining;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.HyperparameterOptimization;
using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.LossFunctions;
using AiDotNet.MixedPrecision;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Regression;
using AiDotNet.Regularization;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// Issue #2554: the streaming supervised path must act on the training configuration it is given. Each test here
/// fails if the configuration it names were dropped: a one-rank distributed backend against no backend, mixed
/// precision observed inside the forward pass, a backend supplied only through the sharding configuration,
/// regularization, and the options that only the in-memory path can honour.
/// </summary>
[Collection("SequentialDistributedStreamingEnablers")]
public sealed class StreamingTrainingConfigurationTests
{
    private const int InDim = 8;
    private const int HiddenDim = 16;
    private const int OutDim = 4;

    /// <summary>
    /// A dense layer that records, on every training forward pass, whether each of its weights is a value FP16 can
    /// hold exactly. That is true only while mixed precision has the low-precision working copy in place.
    /// </summary>
    private sealed class PrecisionProbeDenseLayer : DenseLayer<float>
    {
        public PrecisionProbeDenseLayer(int outputSize)
            : base(outputSize: outputSize, activationFunction: new ReLUActivation<float>())
        {
        }

        public bool Recording { get; set; }

        public int ForwardCalls { get; private set; }

        public int ForwardCallsWithEveryWeightInHalfPrecision { get; private set; }

        protected override Tensor<float> ForwardTraced(Tensor<float> input)
        {
            if (Recording)
            {
                ForwardCalls++;
                if (GetTrainableParameters().All(IsHeldExactlyByHalf))
                {
                    ForwardCallsWithEveryWeightInHalfPrecision++;
                }
            }

            return base.ForwardTraced(input);
        }
    }

    private static bool IsHeldExactlyByHalf(Tensor<float> tensor)
    {
        ReadOnlySpan<float> values = tensor.Data.Span;
        for (int i = 0; i < values.Length; i++)
        {
            if (!IsHalfValue(values[i]))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Whether an FP32 value is one FP16 can hold exactly, worked out from the bit layout so the check does not
    /// lean on the conversion the library itself uses: a normal FP16 value has an exponent in [-14, 15] and only
    /// the top 10 of FP32's 23 mantissa bits set, and a subnormal one is a whole multiple of 2^-24.
    /// </summary>
    private static bool IsHalfValue(float value)
    {
        if (value == 0f || float.IsNaN(value) || float.IsInfinity(value))
        {
            return true;
        }

        double magnitude = Math.Abs((double)value);
        if (magnitude > 65504.0)
        {
            return false;
        }

        if (magnitude >= 6.103515625e-5)
        {
            int bits = BitConverter.ToInt32(BitConverter.GetBytes(value), 0);
            return (bits & 0x1FFF) == 0;
        }

        double steps = magnitude * 16777216.0;
        return steps == Math.Floor(steps);
    }

    private static (Tensor<float>[] Inputs, Tensor<float>[] Targets) CreateDataset(int sampleCount, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var inputs = new Tensor<float>[sampleCount];
        var targets = new Tensor<float>[sampleCount];
        for (int i = 0; i < sampleCount; i++)
        {
            var x = new float[InDim];
            for (int d = 0; d < InDim; d++)
            {
                x[d] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            var y = new float[OutDim];
            for (int o = 0; o < OutDim; o++)
            {
                float acc = 0.1f * (o + 1);
                for (int d = 0; d < InDim; d++)
                {
                    acc += (((d + o) % 2 == 0) ? 0.65f : -0.45f) * x[d];
                }

                y[o] = acc;
            }

            inputs[i] = new Tensor<float>(new[] { InDim }, new Vector<float>(x));
            targets[i] = new Tensor<float>(new[] { OutDim }, new Vector<float>(y));
        }

        return (inputs, targets);
    }

    private static StreamingDataLoader<float, Tensor<float>, Tensor<float>> CreateLoader(
        Tensor<float>[] inputs,
        Tensor<float>[] targets,
        int batchSize)
    {
        return new StreamingDataLoader<float, Tensor<float>, Tensor<float>>(
            sampleCount: inputs.Length,
            sampleReader: (idx, ct) => Task.FromResult((inputs[idx], targets[idx])),
            batchSize: batchSize,
            prefetchCount: 1,
            numWorkers: 1);
    }

    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> CreateAdam(bool clip = true)
    {
        return new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 50,
                UseAdaptiveLearningRate = false,
                UseEarlyStopping = false,
                Tolerance = 0.0,
                EnableGradientClipping = clip,
                MaxGradientNorm = 0.5,
                FitnessCalculator = new MeanSquaredErrorFitnessCalculator<float, Tensor<float>, Tensor<float>>(),
            });
    }

    private static FeedForwardNeuralNetwork<float> CreateMlp(
        IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> optimizer,
        ILayer<float>? hiddenLayer = null)
    {
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(InDim),
            hiddenLayer ?? new DenseLayer<float>(outputSize: HiddenDim, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(outputSize: OutDim, activationFunction: new IdentityActivation<float>()),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: InDim,
            outputSize: OutDim,
            layers: layers);
        var network = new FeedForwardNeuralNetwork<float>(
            architecture,
            optimizer: optimizer,
            lossFunction: new MeanSquaredErrorLoss<float>());
        _ = network.Predict(new Tensor<float>(new[] { 1, InDim }));
        return network;
    }

    /// <summary>Initial weights that FP16 cannot hold exactly, so a low-precision copy is distinguishable.</summary>
    private static Vector<float> CreateInitialWeights(int length, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var values = new float[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = (float)(rng.NextDouble() * 0.8 - 0.4) + 1.0e-6f;
        }

        return new Vector<float>(values);
    }

    private static double MaxAbsDiff(Vector<float> a, Vector<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        double max = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            Assert.False(float.IsNaN(a[i]) || float.IsInfinity(a[i]), $"Non-finite value at index {i}: {a[i]}");
            Assert.False(float.IsNaN(b[i]) || float.IsInfinity(b[i]), $"Non-finite value at index {i}: {b[i]}");
            max = Math.Max(max, Math.Abs((double)a[i] - b[i]));
        }

        return max;
    }

    private static double Norm(Vector<float> v)
    {
        double sum = 0.0;
        for (int i = 0; i < v.Length; i++)
        {
            sum += (double)v[i] * v[i];
        }

        return Math.Sqrt(sum);
    }

    private static GpuAccelerationConfig CpuOnlyConfig() => new GpuAccelerationConfig
    {
        DeviceType = GpuDeviceType.CPU,
        UsageLevel = GpuUsageLevel.AlwaysCpu,
    };

    private static StreamingTrainingOptions<float, Tensor<float>, Tensor<float>> StepOptions(long maxSteps, int seed = 17) =>
        new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
        {
            Seed = seed,
            MaxSteps = maxSteps,
        };

    private static void Shutdown(IEnumerable<ICommunicationBackend<float>> backends)
    {
        foreach (var backend in backends)
        {
            if (backend.IsInitialized)
            {
                backend.Shutdown();
            }
        }
    }

    [Theory(Timeout = 90000)]
    [InlineData(DistributedStrategy.DDP, 1)]
    [InlineData(DistributedStrategy.ZeRO1, 1)]
    [InlineData(DistributedStrategy.ZeRO2, 1)]
    [InlineData(DistributedStrategy.FSDP, 1)]
    [InlineData(DistributedStrategy.DDP, 2)]
    public async Task NeuralNetwork_OneRankBackend_TrainsExactlyLikeNoBackend(DistributedStrategy strategy, int accumulationSteps)
    {
        var (inputs, targets) = CreateDataset(sampleCount: 20, seed: 301);
        const long steps = 7;

        async Task<Vector<float>> TrainAsync(ICommunicationBackend<float>? backend)
        {
            var optimizer = CreateAdam();
            var model = CreateMlp(optimizer);
            model.SetParameters(CreateInitialWeights(model.GetParameters().Length, seed: 302));
            var options = StepOptions(steps);
            options.GradientAccumulationSteps = accumulationSteps;

            var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 6))
                .ConfigureGpuAcceleration(CpuOnlyConfig())
                .ConfigureStreamingTraining(options);
            if (backend is not null)
            {
                builder = builder.ConfigureDistributedTraining(backend, strategy);
            }

            await builder.BuildAsync();
            return model.GetParameters();
        }

        Vector<float> expected = await TrainAsync(backend: null);

        var oneRank = new InMemoryCommunicationBackend<float>(rank: 0, worldSize: 1, "one_rank_" + Guid.NewGuid().ToString("N"));
        Vector<float> actual;
        try
        {
            actual = await TrainAsync(oneRank);
        }
        finally
        {
            Shutdown(new[] { oneRank });
        }

        double diff = MaxAbsDiff(expected, actual);
        Assert.True(diff == 0.0, $"A one-rank {strategy} backend changed the training result by {diff:R}.");
    }

    [Fact(Timeout = 90000)]
    public async Task DistributedTraining_BackendGivenOnlyThroughTheShardingConfiguration_SynchronizesTheRanks()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 24, seed: 311);
        const long steps = 6;

        var referenceOptimizer = CreateAdam();
        var reference = CreateMlp(referenceOptimizer);
        Vector<float> initial = CreateInitialWeights(reference.GetParameters().Length, seed: 312);
        reference.SetParameters(initial.Clone());
        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(reference)
            .ConfigureOptimizer(referenceOptimizer)
            .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 8))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureStreamingTraining(StepOptions(steps))
            .BuildAsync();

        string session = "cfg_only_" + Guid.NewGuid().ToString("N");
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
                model.SetParameters(initial.Clone());
                models[rank] = model;

                await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                    .ConfigureModel(model)
                    .ConfigureOptimizer(optimizer)
                    .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 4))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureDistributedTraining(
                        backend: null,
                        strategy: DistributedStrategy.DDP,
                        configuration: new ShardingConfiguration<float>(backends[rank]))
                    .ConfigureStreamingTraining(StepOptions(steps))
                    .BuildAsync();
            })));

            // Each rank reads half of every global batch. Ranks that ignored the configuration would each train on
            // their own batches of 4 and match neither each other's data nor the batch-of-8 reference.
            Assert.True(MaxAbsDiff(models[0].GetParameters(), models[1].GetParameters()) < 1e-6, "The ranks diverged.");
            double diff = MaxAbsDiff(reference.GetParameters(), models[0].GetParameters());
            Assert.True(diff < 1e-5, $"Two ranks differ from one process at the same global batch by {diff:R}.");
        }
        finally
        {
            Shutdown(backends);
        }
    }

    [Theory(Timeout = 90000)]
    [InlineData(MixedPrecisionType.FP16)]
    [InlineData(MixedPrecisionType.BF16)]
    public async Task MixedPrecision_OnStreaming_RunsTheForwardInLowPrecisionScalesTheLossAndKeepsFullPrecisionMasters(
        MixedPrecisionType precision)
    {
        var (inputs, targets) = CreateDataset(sampleCount: 24, seed: 321);
        const long steps = 6;
        const double initialScale = 512.0;
        Vector<float>? startingWeights = null;

        async Task<(FeedForwardNeuralNetwork<float> Model, PrecisionProbeDenseLayer Probe, AiModelResult<float, Tensor<float>, Tensor<float>> Result)>
            TrainAsync(bool mixedPrecision)
        {
            // Plain SGD: its step is proportional to the gradient, so a loss scale that was never divided back
            // out would move the weights hundreds of times too far. Adam's step would hide that.
            var optimizer = new StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(
                null,
                new StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 0.05,
                    MaxIterations = 50,
                    UseAdaptiveLearningRate = false,
                    UseEarlyStopping = false,
                    Tolerance = 0.0,
                    EnableGradientClipping = false,
                    FitnessCalculator = new MeanSquaredErrorFitnessCalculator<float, Tensor<float>, Tensor<float>>(),
                });
            var probe = new PrecisionProbeDenseLayer(HiddenDim);
            var model = CreateMlp(optimizer, probe);
            startingWeights = CreateInitialWeights(model.GetParameters().Length, seed: 322);
            model.SetParameters(startingWeights.Clone());
            Assert.False(probe.GetTrainableParameters().All(IsHeldExactlyByHalf),
                "The starting weights must not already be FP16 values, or the probe could not tell the two apart.");

            var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 4))
                .ConfigureGpuAcceleration(CpuOnlyConfig())
                .ConfigureStreamingTraining(StepOptions(steps));
            if (mixedPrecision)
            {
                builder = builder.ConfigureMixedPrecision(new MixedPrecisionConfig
                {
                    PrecisionType = precision,
                    InitialLossScale = initialScale,
                    EnableDynamicScaling = true,
                    ScaleGrowthInterval = 2,
                    ScaleGrowthFactor = 2.0,
                });
            }

            probe.Recording = true;
            var result = await builder.BuildAsync();
            probe.Recording = false;
            return (model, probe, result);
        }

        var (fullModel, fullProbe, fullResult) = await TrainAsync(mixedPrecision: false);
        var (mixedModel, mixedProbe, mixedResult) = await TrainAsync(mixedPrecision: true);

        // Reported state.
        Assert.False(fullResult.MixedPrecisionEngaged);
        Assert.True(mixedResult.MixedPrecisionEngaged, mixedResult.MixedPrecisionStatus);
        Assert.StartsWith("engaged:", mixedResult.MixedPrecisionStatus, StringComparison.Ordinal);
        Assert.True(mixedModel.IsMixedPrecisionEnabled);

        // The forward pass ran on low-precision weights. BF16 keeps fewer mantissa bits than FP16 but a wider
        // exponent, so for these magnitudes a BF16 value is not in general an FP16 value; FP16 is checked exactly.
        Assert.True(mixedProbe.ForwardCalls >= steps, $"The probe saw {mixedProbe.ForwardCalls} forward passes for {steps} steps.");
        if (precision == MixedPrecisionType.FP16)
        {
            Assert.Equal(mixedProbe.ForwardCalls, mixedProbe.ForwardCallsWithEveryWeightInHalfPrecision);
        }

        // Without the configuration no forward pass ever sees low-precision weights.
        Assert.Equal(0, fullProbe.ForwardCallsWithEveryWeightInHalfPrecision);

        // The loss scaler took part in every optimizer step and grew on schedule, and nothing overflowed.
        LossScaler<float> scaler = (mixedModel.ActiveMixedPrecisionContext
            ?? throw new InvalidOperationException("Mixed precision is enabled but the model has no context.")).LossScaler;
        Assert.Equal((int)steps, scaler.TotalUpdates);
        Assert.Equal(0, scaler.SkippedUpdates);
        Assert.True(scaler.Scale > initialScale, $"The loss scale never grew from {initialScale} (now {scaler.Scale}).");

        // The master weights the optimizer updates stay FP32: after training they are not FP16 values.
        Assert.False(mixedProbe.GetTrainableParameters().All(IsHeldExactlyByHalf),
            "The trained weights are all FP16 values, so the FP32 masters were not restored before the update.");

        // Gradients were unscaled again and the reported loss is the unscaled one: the run tracks the FP32 run
        // closely, and differs from it only by the rounding of the working weights.
        Vector<float> initialWeights = startingWeights
            ?? throw new InvalidOperationException("The training helper did not record the starting weights.");
        double moved = MaxAbsDiff(initialWeights, fullModel.GetParameters());
        double drift = MaxAbsDiff(fullModel.GetParameters(), mixedModel.GetParameters());
        Assert.True(moved > 1e-2, $"The FP32 run barely trained (largest weight change {moved:R}).");
        Assert.True(drift > 0.0, "Mixed precision produced the FP32 weights bit for bit, so it changed nothing.");
        Assert.True(drift < moved / 10.0,
            $"Mixed precision moved the weights {drift:R} away from the FP32 run, which itself moved them {moved:R}.");
        double fullLoss = fullModel.GetLastLoss();
        double mixedLoss = mixedModel.GetLastLoss();
        Assert.True(mixedLoss < fullLoss * 2.0 + 1e-6 && mixedLoss > fullLoss * 0.5 - 1e-6,
            $"The reported loss {mixedLoss:R} is not on the scale of the FP32 loss {fullLoss:R}; a scaled loss leaked out.");
    }

    [Fact(Timeout = 90000)]
    public async Task Regularization_OnStreaming_ReachesTheOptimizerAndShrinksTheWeights()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 24, seed: 331);
        const long steps = 12;

        async Task<(Vector<float> Weights, AdamOptimizer<float, Tensor<float>, Tensor<float>> Optimizer)> TrainAsync(
            IRegularization<float, Tensor<float>, Tensor<float>>? regularization)
        {
            var optimizer = CreateAdam(clip: false);
            var model = CreateMlp(optimizer);
            model.SetParameters(CreateInitialWeights(model.GetParameters().Length, seed: 332));
            var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 8))
                .ConfigureGpuAcceleration(CpuOnlyConfig())
                .ConfigureStreamingTraining(StepOptions(steps));
            if (regularization is not null)
            {
                builder = builder.ConfigureRegularization(regularization);
            }

            await builder.BuildAsync();
            return (model.GetParameters(), optimizer);
        }

        var l2 = new L2Regularization<float, Tensor<float>, Tensor<float>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 5.0 });

        var (plain, _) = await TrainAsync(regularization: null);
        var (regularized, regularizedOptimizer) = await TrainAsync(l2);

        Assert.Same(l2, regularizedOptimizer.ActiveRegularization);
        Assert.True(MaxAbsDiff(plain, regularized) > 1e-3, "The configured regularization did not change training.");
        Assert.True(Norm(regularized) < Norm(plain),
            $"A strong L2 penalty left the weights larger ({Norm(regularized):R}) than no penalty ({Norm(plain):R}).");
    }

    [Fact(Timeout = 90000)]
    public async Task Regularization_OnStreaming_IsRefusedWhenNothingCouldApplyIt()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 8, seed: 341);
        var l2 = new L2Regularization<float, Tensor<float>, Tensor<float>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.5 });

        // A neural network with no optimizer configured on the builder: there is no slot to put the penalty in.
        var optimizer = CreateAdam();
        var network = CreateMlp(optimizer);
        var missingOptimizer = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(network)
                .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 4))
                .ConfigureGpuAcceleration(CpuOnlyConfig())
                .ConfigureRegularization(l2)
                .ConfigureStreamingTraining(StepOptions(1))
                .BuildAsync());
        Assert.Contains("ConfigureRegularization", missingOptimizer.Message, StringComparison.Ordinal);

        // A model that updates itself without an optimizer.
        var rowInputs = new[] { new Matrix<double>(1, 2), new Matrix<double>(1, 2) };
        var rowTargets = new[] { new Vector<double>(new[] { 1.0 }), new Vector<double>(new[] { 2.0 }) };
        var regression = new MultipleRegression<double>();
        regression.SetParameters(new Vector<double>(new[] { 0.1, 0.2, 0.0 }));
        var notOptimizerTrained = await Assert.ThrowsAsync<NotSupportedException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureModel(regression)
                .ConfigureDataLoader(new StreamingDataLoader<double, Matrix<double>, Vector<double>>(
                    sampleCount: rowInputs.Length,
                    sampleReader: (idx, ct) => Task.FromResult((rowInputs[idx], rowTargets[idx])),
                    batchSize: 2,
                    prefetchCount: 1,
                    numWorkers: 1))
                .ConfigureRegularization(new L2Regularization<double, Matrix<double>, Vector<double>>(
                    new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.5 }))
                .BuildAsync());
        Assert.Contains("ConfigureRegularization", notOptimizerTrained.Message, StringComparison.Ordinal);
    }

    public static IEnumerable<object[]> InMemoryOnlyOptions()
    {
        yield return new object[] { "ConfigureTrainingGroups" };
        yield return new object[] { "ConfigureFederatedLearning" };
        yield return new object[] { "ConfigureKnowledgeDistillation" };
        yield return new object[] { "ConfigureDistillationStrategy" };
        yield return new object[] { "ConfigureContinualLearning" };
        yield return new object[] { "ConfigureFineTuning" };
        yield return new object[] { "ConfigureSelfSupervisedLearning" };
        yield return new object[] { "ConfigureCrossValidation" };
        yield return new object[] { "ConfigureHyperparameterOptimizer" };
        yield return new object[] { "ConfigureLoRA" };
    }

    [Theory(Timeout = 90000)]
    [MemberData(nameof(InMemoryOnlyOptions))]
    public async Task InMemoryOnlyTrainingOption_OnStreaming_IsRefusedInsteadOfDropped(string option)
    {
        var (inputs, targets) = CreateDataset(sampleCount: 8, seed: 351);
        var optimizer = CreateAdam();
        var model = CreateMlp(optimizer);
        Vector<float> before = model.GetParameters();

        IAiModelBuilder<float, Tensor<float>, Tensor<float>> builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 4))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureStreamingTraining(StepOptions(2));

        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("learning_rate", 1e-4, 1e-2);
        builder = option switch
        {
            "ConfigureTrainingGroups" => builder.ConfigureTrainingGroups(new[] { new[] { 0, 1 }, new[] { 2, 3 } }),
            "ConfigureFederatedLearning" => builder.ConfigureFederatedLearning(new FederatedLearningOptions()),
            "ConfigureKnowledgeDistillation" => builder.ConfigureKnowledgeDistillation(),
            "ConfigureDistillationStrategy" => builder.ConfigureDistillationStrategy(
                DistillationStrategyFactory<float>.CreateResponseBasedStrategy(2.0, 0.0)),
            "ConfigureContinualLearning" => builder.ConfigureContinualLearning(),
            "ConfigureFineTuning" => builder.ConfigureFineTuning(
                new FineTuningConfiguration<float, Tensor<float>, Tensor<float>> { Enabled = true }),
            "ConfigureSelfSupervisedLearning" => builder.ConfigureSelfSupervisedLearning(
                null, (pretrained, _, _) => Task.FromResult(pretrained)),
            "ConfigureCrossValidation" => builder.ConfigureCrossValidation(
                new KFoldCrossValidator<float, Tensor<float>, Tensor<float>>(new CrossValidationOptions { NumberOfFolds = 2 })),
            "ConfigureHyperparameterOptimizer" => builder.ConfigureHyperparameterOptimizer(
                new RandomSearchOptimizer<float, Tensor<float>, Tensor<float>>(maximize: false, seed: 1), searchSpace, nTrials: 1),
            "ConfigureLoRA" => builder.ConfigureLoRA(
                new AiDotNet.LoRA.DefaultLoRAConfiguration<float>(rank: 4, alpha: 4, freezeBaseLayer: true)),
            _ => throw new ArgumentOutOfRangeException(nameof(option), option, "Unknown option."),
        };

        var refused = await Assert.ThrowsAsync<NotSupportedException>(() => builder.BuildAsync());
        Assert.Contains(option, refused.Message, StringComparison.Ordinal);
        Assert.True(MaxAbsDiff(before, model.GetParameters()) == 0.0, "The refused build still trained the model.");
    }
}
