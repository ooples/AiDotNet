// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// A network's first compiled training plan holds its optimizer moments, so it has to stay the plan (ooples/AiDotNet
/// #2383). It was replaced on the second step whenever the trainable-layer set grew during the first: the network's
/// shape walk stopped at the first activation after a convolution, so DenseNet's transition convolutions and
/// MobileNet's inverted-residual stages were only built by the first forward.
/// </summary>
// Sets AiDotNetEngine.Current, a process-wide static: run apart from every other test class. On the CPU engine the
// seeded trajectories below are bit-repeatable; on a GPU engine they are not.
[Collection("EngineCurrentGlobalState")]
public class FusedPlanLifetimeTests
{
    private const int Size = 32, Batch = 2, Seed = 2383;

    /// <summary>The families whose layer set grew on the first forward.</summary>
    public enum LazyStackFamily
    {
        DenseNet121,
        MobileNetV2,
        MobileNetV3,
    }

    private static int ClassesOf(LazyStackFamily family) => family == LazyStackFamily.MobileNetV3 ? 100 : 10;

    private static NeuralNetworkBase<float> Create(LazyStackFamily family)
    {
        int classes = ClassesOf(family);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: Size, inputWidth: Size, inputDepth: 3, outputSize: classes)
        {
            RandomSeed = Seed,
        };
        return family switch
        {
            LazyStackFamily.DenseNet121 => new DenseNetNetwork<float>(architecture,
                new DenseNetConfiguration(DenseNetVariant.DenseNet121, classes, Size, Size)),
            LazyStackFamily.MobileNetV2 => new MobileNetV2Network<float>(architecture,
                new MobileNetV2Configuration(MobileNetV2WidthMultiplier.Alpha035, classes,
                    inputHeight: Size, inputWidth: Size, inputChannels: 3)),
            LazyStackFamily.MobileNetV3 => new MobileNetV3Network<float>(architecture,
                new MobileNetV3Configuration(MobileNetV3Variant.Large, classes, inputHeight: Size, inputWidth: Size),
                lossFunction: new MeanSquaredErrorLoss<float>()),
            _ => throw new ArgumentOutOfRangeException(nameof(family), family, "Unknown family."),
        };
    }

    private static (Tensor<float> Images, Tensor<float> Labels) ImageBatch(int classes)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);
        var images = new Tensor<float>(new[] { Batch, 3, Size, Size });
        for (int i = 0; i < images.Length; i++) images[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var labels = new Tensor<float>(new[] { Batch, classes });
        for (int b = 0; b < Batch; b++) labels[b * classes + b] = 1f;
        return (images, labels);
    }

    private static int TrainableLayerCount(NeuralNetworkBase<float> network)
        => TapeTrainingStep<float>.SnapshotTrainableLayerIdentities(network.Layers).Length;

    /// <summary>
    /// With no forward at all the network already has every trainable layer and its full parameter count, its first
    /// plan is the plan of every later step, and its losses are those of an identically seeded twin that ran a
    /// <c>Predict</c> first, and of a second twin that was only ever trained. Before the fix the forwarded twin was
    /// the only one whose moments survived step two, so the third loss differed; and a shape walk that built the late
    /// layers before <c>Train</c> wired the seeds moved the seed of every layer after them, so the first loss differed.
    /// </summary>
    [Theory(Timeout = 600000)]
    [InlineData(LazyStackFamily.DenseNet121)]
    [InlineData(LazyStackFamily.MobileNetV2)]
    [InlineData(LazyStackFamily.MobileNetV3)]
    public async Task TrainedWithoutAPriorForward_IsFullySizedKeepsItsPlanAndFollowsItsForwardedTwin(LazyStackFamily family)
    {
        await Task.Yield();
        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            using var network = Create(family);
            using var reference = Create(family);
            using var onlyTrained = Create(family);
            var (images, labels) = ImageBatch(ClassesOf(family));

            // The shape walk runs here; nothing has been forwarded.
            network.SetTrainingMode(true);
            int layersBeforeAnyForward = TrainableLayerCount(network);
            long parametersBeforeAnyForward = network.ParameterCount;

            reference.Predict(images);
            Assert.Equal(TrainableLayerCount(reference), layersBeforeAnyForward);
            Assert.Equal(reference.ParameterCount, parametersBeforeAnyForward);

            object? firstPlan = null;
            for (int step = 1; step <= 3; step++)
            {
                network.Train(images, labels);
                reference.Train(images, labels);
                onlyTrained.Train(images, labels);

                var plans = CompiledTapeTrainingStep<float>.ConfiguredPlansOf(network);
                Assert.Single(plans);
                firstPlan ??= plans[0];
                Assert.Same(firstPlan, plans[0]);
                Assert.Equal(reference.GetLastLoss(), network.GetLastLoss());
                Assert.Equal(reference.GetLastLoss(), onlyTrained.GetLastLoss());
                Assert.Single(CompiledTapeTrainingStep<float>.ConfiguredPlansOf(onlyTrained));
            }

            Assert.Equal(layersBeforeAnyForward, TrainableLayerCount(network));
            Assert.Equal(parametersBeforeAnyForward, network.ParameterCount);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    /// <summary>
    /// A layer no shape can size: it builds its child on its first real forward. The fused step's own
    /// lazy-initialization forward builds it, and the step has to go on with the layer set that includes it, or the
    /// next step arrives with a set the step never saw and replaces the plan.
    /// </summary>
    [Fact(Timeout = 300000)]
    public async Task LayerBuiltByTheFirstForward_DoesNotReplaceThePlanOnTheSecondStep()
    {
        await Task.Yield();
        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            using var network = BuildSmallNetwork(lateChild: true);
            var (x, y) = SmallBatch();

            network.SetTrainingMode(true);
            int layersBeforeTheFirstStep = TrainableLayerCount(network);
            network.Train(x, y);
            var afterFirstStep = CompiledTapeTrainingStep<float>.ConfiguredPlansOf(network);
            Assert.True(TrainableLayerCount(network) > layersBeforeTheFirstStep,
                "The scenario needs a layer set that grows during the first step.");
            network.Train(x, y);
            network.Train(x, y);
            var afterThirdStep = CompiledTapeTrainingStep<float>.ConfiguredPlansOf(network);

            Assert.Single(afterFirstStep);
            Assert.Single(afterThirdStep);
            Assert.Same(afterFirstStep[0], afterThirdStep[0]);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    /// <summary>
    /// The session only compares the plan it verified by identity. Holding it strongly kept a replaced plan, with
    /// every activation buffer it owns, alive until the next persistence check.
    /// </summary>
    [Fact(Timeout = 300000)]
    public async Task ReplacedPlan_IsNotKeptAliveByTheSession()
    {
        await Task.Yield();
        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            using var network = BuildSmallNetwork(lateChild: false);
            var (x, y) = SmallBatch();
            network.Predict(x);
            var replaced = TrainTwiceThenReplaceThePlan(network, x, y);

            for (int i = 0; i < 3; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            GC.Collect();

            Assert.False(replaced.IsAlive, "The session still holds the plan it verified after that plan was replaced.");
            GC.KeepAlive(network);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference TrainTwiceThenReplaceThePlan(
        NeuralNetworkBase<float> network, Tensor<float> x, Tensor<float> y)
    {
        network.Train(x, y);
        network.Train(x, y);
        var replaced = new WeakReference(CompiledTapeTrainingStep<float>.ConfiguredPlansOf(network)[0]);
        CompiledTapeTrainingStep<float>.Invalidate(network);
        network.Train(x, y);
        Assert.NotSame(replaced.Target, CompiledTapeTrainingStep<float>.ConfiguredPlansOf(network)[0]);
        return replaced;
    }

    /// <summary>
    /// A plan that is still the configured one cannot be collected, so the periodic persistence check keeps being
    /// answered by the identity compare after a collection and never falls back to sampling the parameters.
    /// </summary>
    [Fact(Timeout = 300000)]
    public async Task ConfiguredPlan_KeepsTakingTheStructuralRecheckAfterACollection()
    {
        await Task.Yield();
        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            using var network = BuildSmallNetwork(lateChild: false);
            var (x, y) = SmallBatch();
            network.Train(x, y);
            network.Train(x, y);
            var session = network.FusedSession;
            Assert.True(session.IsCommitted, "The fused step must be the one training the network.");
            int numericBefore = session.NumericPersistenceProbeCount;
            int structuralBefore = session.StructuralPersistenceRecheckCount;
            Assert.True(numericBefore >= 1, "The first verification samples the parameters.");

            for (int i = 0; i < 3; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            for (int step = 0; step < 2 * FusedTrainingSession<float>.PersistenceRecheckInterval + 1; step++)
                network.Train(x, y);

            Assert.True(session.StructuralPersistenceRecheckCount >= structuralBefore + 2,
                $"Two recheck intervals passed but only {session.StructuralPersistenceRecheckCount - structuralBefore} " +
                "checks were answered by the identity compare.");
            Assert.Equal(numericBefore, session.NumericPersistenceProbeCount);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    private static (Tensor<float> X, Tensor<float> Y) SmallBatch()
    {
        var x = new Tensor<float>(new[] { 4, 8 });
        var y = new Tensor<float>(new[] { 4, 3 });
        for (int b = 0; b < 4; b++)
        {
            for (int f = 0; f < 8; f++) x[b, f] = (((b * 8 + f) % 7) - 3) * 0.1f;
            for (int o = 0; o < 3; o++) y[b, o] = (((b * 3 + o) % 5) - 2) * 0.2f;
        }

        return (x, y);
    }

    private static FeedForwardNeuralNetwork<float> BuildSmallNetwork(bool lateChild)
    {
        var layers = new List<ILayer<float>>();
        if (lateChild) layers.Add(new LateChildLayer());
        else layers.Add(new DenseLayer<float>(8));
        layers.Add(new DenseLayer<float>(3));
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 8,
            outputSize: 3,
            layers: layers)
        {
            RandomSeed = Seed,
        };
        var adam = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null, new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 });
        return new FeedForwardNeuralNetwork<float>(architecture, adam, new MeanSquaredErrorLoss<float>());
    }

    /// <summary>Declares its shape at construction, so no shape walk reaches the child it builds on its first forward.</summary>
    [AiDotNet.Attributes.ElementWiseShape]
    private sealed class LateChildLayer : LayerBase<float>
    {
        private DenseLayer<float>? _child;

        internal LateChildLayer() : base(new[] { 8 }, new[] { 8 }) { }

        public override bool SupportsTraining => true;

        protected override Tensor<float> ForwardTraced(Tensor<float> input)
        {
            if (_child is null)
            {
                _child = new DenseLayer<float>(8);
                RegisterSubLayer(_child);
                _child.SetTrainingMode(IsTrainingMode);
            }

            return _child.Forward(input);
        }

        public override void ResetState() => _child?.ResetState();
    }
}
