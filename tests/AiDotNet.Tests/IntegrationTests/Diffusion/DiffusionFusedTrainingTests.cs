using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Diffusion;

/// <summary>
/// Diffusion trains through the shared fused session with each batch's timesteps as a plan input (#1804). With
/// compilation on, every step must run fused, and the weights must land where the eager steps put them.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class DiffusionFusedTrainingTests
{
    // The fused plan takes Adam's beta1, beta2 and epsilon as float, so double training drifts from the eager step
    // by a few 1e-8 of the parameter scale per step; a step at the wrong timestep moves weights by far more.
    private const double RelativeTolerance = 1e-5;
    private const int Steps = 5;

    private sealed record RunResult(Vector<double> Parameters, int FusedSteps, int EagerSteps);

    private static RunResult Train(bool compile)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            var unet = new UNetNoisePredictor<double>(inputChannels: 4, outputChannels: 4, baseChannels: 8,
                channelMultipliers: [1, 2], numResBlocks: 1, attentionResolutions: [1], contextDim: 0, numHeads: 2,
                inputHeight: 8, seed: 42);
            // The model's seed argument seeds only the weights; the timestep and noise draws follow Options.Seed.
            var model = new StableDiffusion15Model<double>(unet: unet, seed: 42,
                options: new AiDotNet.Models.Options.DiffusionModelOptions<double>
                {
                    TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012,
                    BetaSchedule = AiDotNet.Enums.BetaSchedule.ScaledLinear,
                    LearningRate = 0.001, Seed = 5,
                })
            {
                TrainingSampleSpace = AiDotNet.Enums.DiffusionTrainingSampleSpace.Latent,
            };

            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(9);
            var latents = new Tensor<double>([3, 4, 8, 8]);
            for (int i = 0; i < latents.Length; i++) latents[i] = rng.NextDouble() * 2 - 1;
            for (int step = 0; step < Steps; step++) model.Train(latents, latents);

            return new RunResult(model.GetParameters(), model.FusedTrainingStepCount, model.EagerTrainingStepCount);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }

    [Fact(Timeout = 600000)]
    public async Task CompiledTraining_RunsEveryStepFused_AndTakesTheEagerStep()
    {
        await Task.Yield();
        var eager = Train(compile: false);
        var fused = Train(compile: true);

        Assert.Equal(0, eager.FusedSteps);
        Assert.Equal(Steps, eager.EagerSteps);
        Assert.Equal(0, fused.EagerSteps);
        Assert.Equal(Steps, fused.FusedSteps);

        Assert.Equal(eager.Parameters.Length, fused.Parameters.Length);
        double scale = 0.0, worst = 0.0;
        for (int i = 0; i < eager.Parameters.Length; i++)
        {
            scale = Math.Max(scale, Math.Abs(eager.Parameters[i]));
            worst = Math.Max(worst, Math.Abs(eager.Parameters[i] - fused.Parameters[i]));
        }

        Assert.True(worst <= RelativeTolerance * Math.Max(scale, 1.0),
            $"fused diffusion training diverged from eager by {worst:G6} (parameter scale {scale:G6}).");
    }

    [Fact(Timeout = 300000)]
    public async Task EagerBatchedTraining_ChangesThePrediction()
    {
        // The base batched prediction copied each element through host spans, which cut the tape: a batched Train
        // computed no gradient for any weight, and the model after three steps predicted exactly what it did before.
        await Task.Yield();
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
            var unet = new UNetNoisePredictor<double>(inputChannels: 4, outputChannels: 4, baseChannels: 8,
                channelMultipliers: [1, 2], numResBlocks: 1, attentionResolutions: [1], contextDim: 0, numHeads: 2,
                inputHeight: 8, seed: 42);
            var model = new StableDiffusion15Model<double>(unet: unet, seed: 42)
            {
                TrainingSampleSpace = AiDotNet.Enums.DiffusionTrainingSampleSpace.Latent,
            };
            var probe = new Tensor<double>([1, 4, 8, 8]);
            for (int i = 0; i < probe.Length; i++) probe[i] = Math.Sin(i * 0.37);
            var batch = new Tensor<double>([3, 4, 8, 8]);
            for (int i = 0; i < batch.Length; i++) batch[i] = Math.Cos(i * 0.11);

            var before = model.PredictNoise(probe, 400).ToArray();
            model.Train(batch, batch);
            var after = model.PredictNoise(probe, 400).ToArray();

            double moved = 0.0;
            for (int i = 0; i < before.Length; i++) moved = Math.Max(moved, Math.Abs(before[i] - after[i]));
            Assert.True(moved > 1e-6, $"a batched training step left the prediction unchanged (max change {moved:G3}).");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
        }
    }
}