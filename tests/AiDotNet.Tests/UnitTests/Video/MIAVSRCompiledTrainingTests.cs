using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video;

/// <summary>
/// MIA-VSR under fused compiled training. Its training graph draws fresh Gumbel noise every step and
/// aligns neighbouring frames by data-dependent window shifts from SPyNet flow, so a replay that froze
/// the first step's noise or shifts would train a different model. Compiled training must match the
/// eager tape step for step.
/// </summary>
[Collection("FusedTrainingSerial")]
public class MIAVSRCompiledTrainingTests
{
    private const int Frames = 3, Channels = 3, Side = 8;

    private static MIAVSR<double> CreateModel(SpyNetLayer<double>? flow) => new(
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: Frames, inputDepth: Channels, inputHeight: Side, inputWidth: Side, outputSize: 4),
        new MIAVSROptions
        {
            NumFeatures = 8, NumHeads = 2, WindowSize = 4, FeedForwardRatio = 2,
            NumPropagationBranches = 2, BlocksPerBranch = 1, ScaleFactor = 2,
            ReconstructionChannels = 8, Seed = 1234
        }, flowEstimator: flow);

    private static Tensor<double> Clip(int seed, int side, int channels)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var clip = new Tensor<double>(new[] { 1, Frames, channels, side, side });
        for (int i = 0; i < clip.Length; i++) clip[i] = rng.NextDouble();
        return clip;
    }

    private static double[] TrainSteps(MIAVSR<double> model, bool compiled, out long fusedSteps)
    {
        var previous = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compiled });
            CompiledTapeTrainingStep<double>.Invalidate();
            CompiledTapeTrainingStep<double>.ResetFusedStepCount();
            for (int step = 0; step < 3; step++)
            {
                // A different clip each step: a replay that froze the first step's flow shifts or noise
                // would train on the wrong alignment from step two on.
                model.Train(Clip(10 + step, Side, Channels), Clip(20 + step, Side * 2, Channels));
            }

            fusedSteps = CompiledTapeTrainingStep<double>.GetFusedStepCount();
            return model.GetParameters().ToArray();
        }
        finally
        {
            CompiledTapeTrainingStep<double>.Invalidate();
            CompiledTapeTrainingStep<double>.ResetFusedStepCount();
            TensorCodecOptions.SetCurrent(previous);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompiledTraining_MatchesTheEagerTape(bool pretrainedFlow)
    {
        var compiledModel = CreateModel(pretrainedFlow ? new SpyNetLayer<double>() : null);
        var eagerModel = CreateModel(pretrainedFlow ? new SpyNetLayer<double>() : null);
        compiledModel.Predict(Clip(1, Side, Channels));
        eagerModel.Predict(Clip(1, Side, Channels));
        var initial = compiledModel.GetParameters();
        eagerModel.UpdateParameters(initial);
        var start = initial.ToArray();

        var compiled = TrainSteps(compiledModel, compiled: true, out long fusedSteps);
        var eager = TrainSteps(eagerModel, compiled: false, out _);

        Assert.Equal(3, fusedSteps);
        Assert.Equal(eager.Length, compiled.Length);
        double moved = start.Zip(eager, (a, b) => Math.Abs(a - b)).Max();
        double divergence = compiled.Zip(eager, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(moved > 1e-8, $"Training did not move the weights; parity would be vacuous (max {moved:E3}).");
        Assert.True(divergence < 1e-9, $"Compiled MIA-VSR training diverged from the eager tape by {divergence:E3}.");
    }
}