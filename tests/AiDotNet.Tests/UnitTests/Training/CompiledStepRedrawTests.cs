using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio;
using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// Shared layer code that draws per-step randomness or reads per-step statistics inside a training forward. The fused
/// training path traces the forward once and replays the plan, so a mask built with a host loop, or a statistic read
/// back to the host, is frozen into the plan as the traced step's value: every later step trains on the first step's
/// mask or normalization. Each test trains a fused and an eager copy of the same seeded model side by side on fresh
/// batches and requires the fused loss to follow the eager loss step for step; a frozen mask or statistic drops or
/// scales different elements from step 2 on and the arms diverge. The runs are in double, where the arms agree to the
/// compiled plan's own rounding (well under the 1e-6 relative tolerance) and frozen state moves the loss by percent.
/// </summary>
public class CompiledStepRedrawTests
{
    private const int Steps = 4;
    private const double Tolerance = 1e-6;

    /// <summary>
    /// Swin's stochastic depth drops whole residual branches per sample. With a rate of one half, four samples and two
    /// branches per step, the eager steps draw different masks (asserted), so a plan replaying the traced mask cannot
    /// follow them.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Swin_drop_path_is_redrawn_on_every_compiled_step()
    {
        await Task.Yield();
        const int batch = 4, tokens = 4, dim = 8;
        SwinTransformerBlockLayer<double> Build() => new(dim, numHeads: 2, windowSize: 2, mlpRatio: 2, dropPathRate: 0.5)
        {
            RandomSeed = 17,
        };
        var fused = Build();
        var eager = Build();
        var warm = new Tensor<double>(new[] { batch, tokens, dim });
        fused.Forward(warm);
        eager.Forward(warm);
        eager.SetParameters(fused.GetParameters());
        fused.SetTrainingMode(true);
        eager.SetTrainingMode(true);

        var eagerLosses = TrainSideBySide(new[] { fused }, fused.Forward, new[] { eager }, eager.Forward,
            data => (Random(data, batch, tokens, dim), Random(data, batch, tokens, dim)));
        Assert.True(eagerLosses.Distinct().Count() == eagerLosses.Count, "the eager steps should differ.");

        // The masks themselves: two drop-path forwards of the same layer draw different masks.
        var probe = Build();
        probe.Forward(warm);
        probe.SetParameters(fused.GetParameters());
        probe.SetTrainingMode(true);
        var sample = Random(RandomHelper.CreateSeededRandom(9), batch, tokens, dim);
        Assert.NotEqual(probe.Forward(sample).ToArray(), probe.Forward(sample).ToArray());
    }

    /// <summary>
    /// Wav2Vec2's dropouts, its SpecAugment time mask and its LayerDrop all draw per step. The time mask and LayerDrop
    /// used to be host decisions: a bool mask written into a constant, and a branch that left a dropped layer out of
    /// the traced graph altogether.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Wav2Vec2_dropout_time_mask_and_layer_drop_are_redrawn_on_every_compiled_step()
    {
        await Task.Yield();
        const int batch = 2, samples = 400;
        var configuration = new Wav2Vec2Configuration
        {
            ConvDims = new[] { 8, 8, 8 }, ConvKernels = new[] { 10, 3, 2 }, ConvStrides = new[] { 5, 2, 2 },
            HiddenSize = 16, Layers = 2, Heads = 2, IntermediateSize = 32, PositionalConvKernel = 8, PositionalConvGroups = 4,
            FeatureProjectionDropout = 0.2, HiddenDropout = 0.2, AttentionDropout = 0.2, ActivationDropout = 0.2,
            LayerDrop = 0.5, MaskTimeProbability = 0.3, MaskTimeLength = 2, MaskTimeMinMasks = 1,
        };
        (Wav2Vec2Network<double> Network, List<LayerBase<double>> Layers) Build()
        {
            var layers = new List<LayerBase<double>>();
            var network = new Wav2Vec2Network<double>(AiDotNetEngine.Current, layers, configuration);
            network.InitializeLikeReference(RandomHelper.CreateSeededRandom(3));
            return (network, layers);
        }
        var fused = Build();
        var eager = Build();
        var fusedRandom = RandomHelper.CreateSeededRandom(29);
        var eagerRandom = RandomHelper.CreateSeededRandom(29);
        int frames = fused.Network.Frames(samples);

        var eagerLosses = TrainSideBySide(
            fused.Layers, x => fused.Network.Forward(x, training: true, fusedRandom).Hidden,
            eager.Layers, x => eager.Network.Forward(x, training: true, eagerRandom).Hidden,
            data => (Random(data, batch, samples), Random(data, batch, frames, configuration.HiddenSize)));
        Assert.True(eagerLosses.Distinct().Count() == eagerLosses.Count, "the eager steps should differ.");
    }

    /// <summary>
    /// The waveform standardization Wav2Vec2 language identification applies inside its training forward. Its scale,
    /// mean and spread used to be read back to the host, so a plan reused the traced batch's statistics for every
    /// batch. Each step's batch here has a different level and spread, so stale statistics misnormalize it.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Standardize_recomputes_its_statistics_on_every_compiled_step()
    {
        await Task.Yield();
        const int batch = 2, samples = 96, outputs = 3;
        DenseLayer<double> Build() => new(outputs);
        var fused = Build();
        var eager = Build();
        var warm = new Tensor<double>(new[] { batch, samples });
        fused.Forward(warm);
        eager.Forward(warm);
        eager.SetParameters(fused.GetParameters());
        var engine = AiDotNetEngine.Current;
        int step = 0;

        TrainSideBySide(
            new[] { fused }, x => fused.Forward(AudioEngineOps.Standardize(engine, x, 1e-7)),
            new[] { eager }, x => eager.Forward(AudioEngineOps.Standardize(engine, x, 1e-7)),
            data =>
            {
                double level = 0.5 * step, spread = 1.0 + step++;
                var input = Random(data, batch, samples);
                for (int i = 0; i < input.Length; i++) input[i] = level + spread * input[i];
                return (input, Random(data, batch, outputs));
            });
    }

    /// <summary>The eager losses, after requiring the fused arm to engage and to follow them step for step.</summary>
    private static List<double> TrainSideBySide(
        IReadOnlyList<ITrainableLayer<double>> fusedLayers, Func<Tensor<double>, Tensor<double>> fusedForward,
        IReadOnlyList<ITrainableLayer<double>> eagerLayers, Func<Tensor<double>, Tensor<double>> eagerForward,
        Func<System.Random, (Tensor<double> Input, Tensor<double> Target)> batchOf)
    {
        var stepper = new TapeTrainingStepper<double>(new object());
        var fusedOptimizer = Sgd();
        var eagerOptimizer = Sgd();
        var data = RandomHelper.CreateSeededRandom(5);
        var eagerLosses = new List<double>();
        for (int step = 0; step < Steps; step++)
        {
            var (input, target) = batchOf(data);
            double fused = stepper.Step(Request(fusedLayers, fusedForward, input, target, fusedOptimizer));
            Assert.True(stepper.LastStepFused, $"step {step}: fused path did not engage ({stepper.Session.LastMissReason}).");
            double eager = TapeTrainingStepper<double>.EagerStep(Request(eagerLayers, eagerForward, input, target, eagerOptimizer));
            Assert.True(
                Math.Abs(fused - eager) <= Tolerance * Math.Max(1.0, Math.Abs(eager)),
                $"step {step}: fused loss {fused:R} differs from eager loss {eager:R}; the plan replays stale per-step state.");
            eagerLosses.Add(eager);
        }
        return eagerLosses;
    }

    private static FusedTrainingStepRequest<double> Request(IReadOnlyList<ITrainableLayer<double>> layers,
        Func<Tensor<double>, Tensor<double>> forward, Tensor<double> input, Tensor<double> target,
        IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> optimizer) => new()
    {
        Layers = layers,
        Input = input,
        Target = target,
        Forward = forward,
        ComputeLoss = (predicted, expected) =>
        {
            var engine = AiDotNetEngine.Current;
            var difference = engine.TensorSubtract(predicted, expected);
            return engine.TensorMultiplyScalar(
                engine.ReduceSum(engine.TensorMultiply(difference, difference), null), 1.0 / difference.Length);
        },
        Optimizer = optimizer,
    };

    private static StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>> Sgd()
        => new(model: null, options: new StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 0.05,
            UseAdaptiveLearningRate = false,
        });

    private static Tensor<double> Random(System.Random random, params int[] shape)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = random.NextDouble() * 2.0 - 1.0;
        return tensor;
    }
}
