using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.AutoML;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML;

/// <summary>
/// The model a diffusion AutoML trial builds trains, samples and is scored as a latent diffusion model (#2155).
/// </summary>
/// <remarks>
/// Its Train computed the denoising loss, discarded it, and stepped on an SPSA estimate taken through the whole
/// sampling loop. Its guidance multiplied the noise prediction by the scale, and every trial built a U-Net with a
/// DDIM or PNDM scheduler whatever it asked for. It had no tests; the generator's exclusion cited integration
/// tests that did not exist.
/// </remarks>
public class DiffusionAutoMLTrainingTests
{
    private static DiffusionTrialConfig<double> SmallConfig(
        NoisePredictorType predictor = NoisePredictorType.DiT,
        DiffusionSchedulerType scheduler = DiffusionSchedulerType.DDIM,
        double guidanceScale = 1.0) => new DiffusionTrialConfig<double>
    {
        NoisePredictorType = predictor,
        SchedulerType = scheduler,
        BaseChannels = 16,
        NumHeads = 4,
        TransformerDepth = 2,
        NumResBlocks = 1,
        LatentDim = 4,
        LatentHeight = 2,
        LatentWidth = 2,
        ConditioningDim = 3,
        ImageChannels = 3,
        InferenceSteps = 3,
        GuidanceScale = guidanceScale,
        LearningRate = 1e-3,
    };

    private static Tensor<double> Random(int[] shape, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = rng.NextDouble() * 2.0 - 1.0;
        return tensor;
    }

    private static double[] Values(Vector<double> vector) => Enumerable.Range(0, vector.Length).Select(i => vector[i]).ToArray();

    [Fact(Timeout = 120000)]
    public async Task BuildsThePredictorAndSchedulerTheTrialNames()
    {
        await Task.Yield();
        var dit = new DiffusionAutoMLModel<double>(SmallConfig(NoisePredictorType.DiT, DiffusionSchedulerType.DPMSolver), seed: 1);
        Assert.IsType<DiTNoisePredictor<double>>(dit.NoisePredictor);
        Assert.IsType<DPMSolverMultistepScheduler<double>>(dit.Scheduler);

        var uvit = new DiffusionAutoMLModel<double>(SmallConfig(NoisePredictorType.UViT, DiffusionSchedulerType.Euler), seed: 1);
        Assert.IsType<UViTNoisePredictor<double>>(uvit.NoisePredictor);
        Assert.IsType<EulerDiscreteScheduler<double>>(uvit.Scheduler);

        var unet = new DiffusionAutoMLModel<double>(SmallConfig(NoisePredictorType.UNet, DiffusionSchedulerType.DDPM), seed: 1);
        Assert.IsType<UNetNoisePredictor<double>>(unet.NoisePredictor);
        Assert.IsType<DDPMScheduler<double>>(unet.Scheduler);
    }

    [Fact(Timeout = 120000)]
    public async Task Train_OnImages_MovesTheDenoiserAndLeavesTheAutoencoderFixed()
    {
        await Task.Yield();
        var model = new DiffusionAutoMLModel<double>(SmallConfig(), seed: 3);
        var conditions = Random(new[] { 2, 3 }, 1);
        var images = Random(new[] { 2, 3, 16, 16 }, 2);
        model.Train(conditions, images);

        var denoiserBefore = Values(model.NoisePredictor.GetParameters());
        var autoencoderBefore = Values(((StandardVAE<double>)model.VAE).GetParameters());
        model.Train(conditions, images);
        var denoiserAfter = Values(model.NoisePredictor.GetParameters());
        var autoencoderAfter = Values(((StandardVAE<double>)model.VAE).GetParameters());

        Assert.True(denoiserBefore.Zip(denoiserAfter, (a, b) => a != b).Any(changed => changed),
            "A training step on images left every denoiser weight where it was.");
        // Rombach et al. 2022 train the denoiser on the latents of a fixed autoencoder.
        Assert.Equal(autoencoderBefore, autoencoderAfter);
    }

    [Fact(Timeout = 120000)]
    public async Task TrainAutoencoder_MovesTheAutoencoder()
    {
        await Task.Yield();
        var model = new DiffusionAutoMLModel<double>(SmallConfig(), seed: 4);
        var images = Random(new[] { 1, 3, 16, 16 }, 9);
        model.TrainAutoencoder(images);

        var before = Values(((StandardVAE<double>)model.VAE).GetParameters());
        model.TrainAutoencoder(images);
        var after = Values(((StandardVAE<double>)model.VAE).GetParameters());

        Assert.True(before.Zip(after, (a, b) => a != b).Any(changed => changed),
            "An autoencoder training step left every autoencoder weight where it was.");
    }

    [Fact(Timeout = 120000)]
    public async Task GenerateConditioned_ReturnsOneFiniteImagePerCondition()
    {
        await Task.Yield();
        var model = new DiffusionAutoMLModel<double>(SmallConfig(), seed: 5);
        var images = model.GenerateConditioned(Random(new[] { 2, 3 }, 6), seed: 11);

        Assert.Equal(new[] { 2, 3, 16, 16 }, images.Shape.ToArray());
        for (int i = 0; i < images.Length; i++)
            Assert.False(double.IsNaN(images[i]) || double.IsInfinity(images[i]), $"image value {i} is {images[i]}.");
    }

    [Fact(Timeout = 120000)]
    public async Task Guidance_ChangesTheSampleFromTheSameNoise()
    {
        await Task.Yield();
        // Identical seeds give identical weights and identical training; only the guidance scale differs. The
        // denoisers are trained first because DiT starts from adaLN-Zero, which ignores the condition at init.
        var condition = Random(new[] { 1, 3 }, 7);
        var images = Random(new[] { 1, 3, 16, 16 }, 8);

        Tensor<double> Sample(double guidanceScale)
        {
            var model = new DiffusionAutoMLModel<double>(SmallConfig(guidanceScale: guidanceScale), seed: 6);
            for (int step = 0; step < 3; step++) model.Train(condition, images);
            return model.GenerateConditioned(condition, seed: 12);
        }

        var plain = Sample(1.0);
        var guided = Sample(5.0);
        Assert.Contains(Enumerable.Range(0, plain.Length), i => Math.Abs(plain[i] - guided[i]) > 1e-9);
    }

    [Fact(Timeout = 300000)]
    public async Task Search_TrainsAndScoresATrialOnTheData()
    {
        var automl = new DiffusionAutoML<double>(seed: 1)
        {
            AutoencoderTrainingIterations = 1,
            DiffusionTrainingIterations = 1,
            TrialLimit = 1,
        };
        automl.NoisePredictorTypesToTry.Clear();
        automl.NoisePredictorTypesToTry.Add(NoisePredictorType.DiT);
        automl.SetSearchSpace(new Dictionary<string, ParameterRange>
        {
            ["InferenceSteps"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 2, MaxValue = 2, Step = 1 },
            ["GuidanceScale"] = new ParameterRange { Type = ParameterType.Float, MinValue = 1.0, MaxValue = 3.0, Step = 0.5 },
            ["LearningRate"] = new ParameterRange { Type = ParameterType.Float, MinValue = 1e-4, MaxValue = 1e-3, UseLogScale = true },
            ["BaseChannels"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 16, MaxValue = 16, Step = 16 },
            ["NumResBlocks"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 1, MaxValue = 1, Step = 1 },
            ["LatentDim"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 4, MaxValue = 4, Step = 4 },
        });

        var conditions = Random(new[] { 2, 3 }, 13);
        var images = Random(new[] { 2, 3, 16, 16 }, 14);
        var best = await automl.SearchAsync(conditions, images, conditions, images, TimeSpan.FromMinutes(4));

        var trial = Assert.Single(automl.GetTrialHistory());
        Assert.True(trial.Success, $"The trial failed: {trial.ErrorMessage}");
        Assert.False(double.IsNaN(trial.Score) || double.IsInfinity(trial.Score), $"The trial scored {trial.Score}.");

        // The sizes a trial cannot search come from the data: a 3-wide condition, 3-channel 16 x 16 images.
        var model = Assert.IsType<DiffusionAutoMLModel<double>>(best);
        Assert.Equal(3, model.Config.ConditioningDim);
        Assert.Equal(3, model.Config.ImageChannels);
        Assert.Equal(2, model.Config.LatentHeight);
        Assert.Equal(2, model.Config.LatentWidth);
    }
}
