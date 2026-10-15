using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.GANs;

/// <summary>
/// The GAN's generator step runs on the generator's fused session (#1804). With compilation on every generator step
/// must run fused and both networks, and the generator's BatchNorm running statistics, must land where the eager
/// iterations put them. The discriminator trains between generator steps, so a plan that kept the discriminator's
/// first weights, or the first batch of noise, would diverge here.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class GanFusedTrainingTests
{
    // The fused plan takes Adam's beta1, beta2 and epsilon as float, so double training drifts from the eager step
    // by a few 1e-8 of the parameter scale per step; a step on stale data moves weights by far more.
    private const double Tolerance = 1e-5;
    private const int Batch = 4;

    private static GenerativeAdversarialNetwork<double> Build()
    {
        var generator = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Generative, inputSize: 8, outputSize: 4,
            layers: new List<ILayer<double>>
            {
                new DenseLayer<double>(16, (IActivationFunction<double>)new LeakyReLUActivation<double>()),
                new BatchNormalizationLayer<double>(),
                new DenseLayer<double>(4, (IActivationFunction<double>)new TanhActivation<double>()),
            });
        var discriminator = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.BinaryClassification, inputSize: 4, outputSize: 1,
            layers: new List<ILayer<double>>
            {
                new DenseLayer<double>(16, (IActivationFunction<double>)new LeakyReLUActivation<double>()),
                new DenseLayer<double>(1, (IActivationFunction<double>)new IdentityActivation<double>()),
            });
        var gan = new GenerativeAdversarialNetwork<double>(generator, discriminator, InputType.OneDimensional);
        // Resolve lazily shaped layers with plain forwards: a warm-up Train would give each copy optimizer moments
        // built from its own initial weights, which SetParameters does not carry to the twin.
        Net(gan.Generator).Predict(Data(8, 0));
        Net(gan.Discriminator).Predict(Data(4, 100));
        return gan;
    }

    private static Tensor<double> Data(int width, int step)
    {
        var tensor = new Tensor<double>([Batch, width]);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = Math.Sin(0.37 * i + 1.3 * step);
        return tensor;
    }

    private static NeuralNetworkBase<double> Net(INeuralNetworkModel<double> model) => (NeuralNetworkBase<double>)model;

    private static double[] RunningStatistics(GenerativeAdversarialNetwork<double> gan)
    {
        var values = new List<double>();
        foreach (var layer in Net(gan.Generator).Layers)
        {
            if (layer is not LayerBase<double> concrete) continue;
            foreach (var (_, buffer) in concrete.GetRegisteredBuffers())
                for (int i = 0; i < buffer.Length; i++) values.Add(buffer[i]);
        }
        return values.ToArray();
    }

    private sealed record RunResult(Vector<double> Generator, Vector<double> Discriminator, double[] Statistics, int FusedSteps, int EagerSteps);

    private static RunResult Run(GenerativeAdversarialNetwork<double> gan, bool compile)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            for (int step = 1; step <= 4; step++) gan.Train(Data(8, step), Data(4, 100 + step));
            var generator = Net(gan.Generator);
            return new RunResult(generator.GetParameters(), Net(gan.Discriminator).GetParameters(), RunningStatistics(gan),
                generator.CustomFusedStepCount, generator.CustomEagerStepCount);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }

    private static double Worst(IReadOnlyList<double> a, IReadOnlyList<double> b)
    {
        Assert.Equal(a.Count, b.Count);
        double worst = 0.0;
        for (int i = 0; i < a.Count; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    [Fact(Timeout = 120000)]
    public async Task GeneratorSteps_RunFused_AndTakeTheEagerIteration()
    {
        await Task.Yield();
        var gan = Build();
        var twin = Build();
        Net(twin.Generator).SetParameters(Net(gan.Generator).GetParameters());
        Net(twin.Discriminator).SetParameters(Net(gan.Discriminator).GetParameters());
        var before = Net(gan.Generator).GetParameters();

        int priorFused = Net(gan.Generator).CustomFusedStepCount, priorEager = Net(gan.Generator).CustomEagerStepCount;
        int twinFused = Net(twin.Generator).CustomFusedStepCount, twinEager = Net(twin.Generator).CustomEagerStepCount;
        var eager = Run(gan, compile: false);
        var fused = Run(twin, compile: true);

        Assert.Equal(priorFused, eager.FusedSteps);
        Assert.Equal(priorEager + 4, eager.EagerSteps);
        Assert.Equal(twinEager, fused.EagerSteps);
        Assert.Equal(twinFused + 4, fused.FusedSteps);

        double moved = Worst(eager.Generator.ToArray(), before.ToArray());
        Assert.True(moved > 100 * Tolerance, $"training barely moved the generator ({moved:G6}).");
        double generator = Worst(eager.Generator.ToArray(), fused.Generator.ToArray());
        double discriminator = Worst(eager.Discriminator.ToArray(), fused.Discriminator.ToArray());
        double statistics = Worst(eager.Statistics, fused.Statistics);
        Assert.True(generator <= Tolerance, $"fused generator diverged from eager by {generator:G6}.");
        Assert.True(discriminator <= Tolerance, $"fused discriminator diverged from eager by {discriminator:G6}.");
        Assert.True(statistics <= Tolerance, $"fused running statistics diverged from eager by {statistics:G6}.");
    }

    [Fact(Timeout = 120000)]
    public async Task GeneratedSamplesForTheDiscriminator_LeaveTheRunningStatisticsToTheGeneratorStep()
    {
        // The discriminator's samples come from a forward that must not advance the generator's running statistics:
        // one iteration advances them exactly as one training-mode forward does.
        await Task.Yield();
        var gan = Build();
        var statisticsBefore = RunningStatistics(gan);
        var generator = Net(gan.Generator);
        generator.ForwardForTrainingDetached(Data(8, 7));
        Assert.Equal(statisticsBefore, RunningStatistics(gan));
    }

    private static Tensor<double> OneHot(int classes, int step)
    {
        var tensor = new Tensor<double>([Batch, classes]);
        for (int r = 0; r < Batch; r++) tensor[r, (r + step) % classes] = 1.0;
        return tensor;
    }

    private static ConditionalGAN<double> BuildConditional() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple, inputSize: 9, outputSize: 4),
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple, inputSize: 4, outputSize: 1),
        numConditionClasses: 3,
        InputType.OneDimensional);

    [Fact(Timeout = 120000)]
    public async Task ConditionalGanGeneratorSteps_RunFused_AndReadEachStepsConditions()
    {
        // The generator's loss slices the class conditions out of the step's input; a plan that read them from the
        // first step's host tensor would train every later step against the first step's classes.
        await Task.Yield();
        using var cgan = BuildConditional();
        using var twin = BuildConditional();
        // Plain forwards resolve the lazy layers without giving either copy optimizer moments of its own.
        foreach (var g in new[] { cgan, twin })
        {
            Net(g.Generator).Predict(Data(9, 0));
            Net(g.Discriminator).Predict(Data(7, 0));
        }
        Net(twin.Generator).SetParameters(Net(cgan.Generator).GetParameters());
        Net(twin.Discriminator).SetParameters(Net(cgan.Discriminator).GetParameters());
        var before = Net(cgan.Generator).GetParameters();

        Vector<double> RunConditional(ConditionalGAN<double> gan, bool compile, out int fused, out int eager)
        {
            var original = TensorCodecOptions.Current;
            var generator = Net(gan.Generator);
            int fused0 = generator.CustomFusedStepCount, eager0 = generator.CustomEagerStepCount;
            try
            {
                TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
                CompiledTapeTrainingStep<double>.Invalidate();
                for (int step = 1; step <= 4; step++) gan.TrainStep(Data(4, 100 + step), OneHot(3, step), Data(6, step));
            }
            finally
            {
                TensorCodecOptions.SetCurrent(original);
                CompiledTapeTrainingStep<double>.Invalidate();
            }
            fused = generator.CustomFusedStepCount - fused0;
            eager = generator.CustomEagerStepCount - eager0;
            return generator.GetParameters();
        }

        var eagerParameters = RunConditional(cgan, compile: false, out int eagerFused, out int eagerEager);
        var fusedParameters = RunConditional(twin, compile: true, out int fusedFused, out int fusedEager);

        Assert.Equal(0, eagerFused);
        Assert.Equal(4, eagerEager);
        Assert.Equal(0, fusedEager);
        Assert.Equal(4, fusedFused);
        double moved = Worst(eagerParameters.ToArray(), before.ToArray());
        Assert.True(moved > 10 * Tolerance, $"training barely moved the generator ({moved:G6}).");
        double worst = Worst(eagerParameters.ToArray(), fusedParameters.ToArray());
        Assert.True(worst <= Tolerance, $"fused conditional generator diverged from eager by {worst:G6}.");
    }

    private static Tensor<float> FloatData(int width, int step)
    {
        var tensor = new Tensor<float>([Batch, width]);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = (float)Math.Sin(0.37 * i + 1.3 * step);
        return tensor;
    }

    [Fact(Timeout = 120000)]
    public async Task WganGeneratorSteps_RunFused_AndTrainTheGenerator()
    {
        // WGAN draws its generator noise from the unseeded engine generator, so two runs cannot be compared; every
        // generator step must still run fused and move the generator. Float: the plan's fused RMSprop (Arjovsky et
        // al. 2017's optimizer) is float-only.
        await Task.Yield();
        var wgan = new WGAN<float>(
            generatorArchitecture: new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 4),
            criticArchitecture: new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression, inputSize: 4, outputSize: 1),
            inputType: InputType.OneDimensional);
        wgan.TrainStep(FloatData(4, 0), FloatData(8, 1));
        var generator = (NeuralNetworkBase<float>)wgan.Generator;
        var before = generator.GetParameters();
        int fused0 = generator.CustomFusedStepCount, eager0 = generator.CustomEagerStepCount;
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();
            for (int step = 1; step <= 3; step++) wgan.TrainStep(FloatData(4, step), FloatData(8, 10 + step));
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<float>.Invalidate();
        }

        Assert.Equal(3, generator.CustomFusedStepCount - fused0);
        Assert.Equal(0, generator.CustomEagerStepCount - eager0);
        var after = generator.GetParameters();
        double moved = 0.0;
        for (int i = 0; i < after.Length; i++) moved = Math.Max(moved, Math.Abs(after[i] - before[i]));
        Assert.True(moved > 0, "the fused generator steps did not move the generator.");
    }}
