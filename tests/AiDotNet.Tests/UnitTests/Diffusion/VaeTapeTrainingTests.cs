using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// VAEs train on the gradient tape (#2155).
/// </summary>
/// <remarks>
/// VAEModelBase.ComputeGradients took its exact path only when SupportsExactGradients was set, and no VAE set it,
/// so all eleven VAEs trained on a three-sample SPSA estimate over every weight at once. The tape-based
/// ComputeGradientsWithTape existed and nothing called it.
/// </remarks>
public class VaeTapeTrainingTests
{
    private static StandardVAE<double> SmallVae(int seed) => new StandardVAE<double>(
        inputChannels: 3,
        latentChannels: 4,
        baseChannels: 8,
        channelMultipliers: new[] { 1, 2 },
        numResBlocksPerLevel: 1,
        seed: seed);

    private static Tensor<double> Image(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var image = new Tensor<double>(new[] { 1, 3, 8, 8 });
        for (int i = 0; i < image.Length; i++) image[i] = rng.NextDouble() * 2.0 - 1.0;
        return image;
    }

    private static double ReconstructionError(StandardVAE<double> vae, Tensor<double> image)
    {
        var reconstruction = vae.Predict(image);
        double sum = 0.0;
        for (int i = 0; i < image.Length; i++)
        {
            double difference = reconstruction[i] - image[i];
            sum += difference * difference;
        }

        return sum / image.Length;
    }

    [Fact(Timeout = 120000)]
    public async Task Train_ReducesTheReconstructionError()
    {
        await Task.Yield();
        var vae = SmallVae(seed: 7);
        vae.TrainingLearningRate = 1e-3;
        var image = Image(1);

        double before = ReconstructionError(vae, image);
        for (int step = 0; step < 40; step++) vae.Train(image, image);
        double after = ReconstructionError(vae, image);

        Assert.True(after < 0.9 * before,
            $"40 Adam steps on one image should cut its reconstruction error by 10%: before {before:E4}, after {after:E4}.");
    }

    [Fact(Timeout = 120000)]
    public async Task ComputeGradients_MatchesCentralDifferencesInParameterOrder()
    {
        await Task.Yield();
        var vae = SmallVae(seed: 3);
        var image = Image(2);
        _ = vae.Predict(image);

        var analytic = vae.ComputeGradients(image, image);
        var parameters = vae.GetParameters();
        Assert.Equal(parameters.Length, analytic.Length);

        // Coordinates spread over the whole vector, so a gradient written in a different order than
        // GetParameters reads cannot match.
        var rng = RandomHelper.CreateSeededRandom(11);
        const double h = 1e-5;
        int checkedCoordinates = 0;
        for (int sample = 0; sample < 24; sample++)
        {
            int index = rng.Next(parameters.Length);
            var plus = parameters.Clone();
            plus[index] += h;
            vae.SetParameters(plus);
            double lossPlus = vae.DefaultLossFunction.CalculateLoss(vae.Predict(image).ToVector(), image.ToVector());

            var minus = parameters.Clone();
            minus[index] -= h;
            vae.SetParameters(minus);
            double lossMinus = vae.DefaultLossFunction.CalculateLoss(vae.Predict(image).ToVector(), image.ToVector());

            vae.SetParameters(parameters);
            double numeric = (lossPlus - lossMinus) / (2.0 * h);
            if (Math.Abs(numeric) < 1e-9 && Math.Abs(analytic[index]) < 1e-9) continue;

            checkedCoordinates++;
            Assert.True(Math.Abs(numeric - analytic[index]) <= 1e-6 + 1e-3 * Math.Abs(numeric),
                $"parameter {index}: tape gradient {analytic[index]:E6} against central difference {numeric:E6}.");
        }

        Assert.True(checkedCoordinates >= 12, $"Only {checkedCoordinates} of 24 sampled coordinates had a gradient.");
    }

    [Fact(Timeout = 120000)]
    public async Task Train_UsesTheConfiguredLearningRate()
    {
        await Task.Yield();
        var image = Image(4);

        double StepSize(double learningRate)
        {
            var vae = SmallVae(seed: 5);
            vae.TrainingLearningRate = learningRate;
            _ = vae.Predict(image);
            var before = vae.GetParameters();
            vae.Train(image, image);
            var after = vae.GetParameters();
            double largest = 0.0;
            for (int i = 0; i < before.Length; i++) largest = Math.Max(largest, Math.Abs(after[i] - before[i]));
            return largest;
        }

        // Adam's first step moves each weight by about the learning rate, whatever its gradient's size.
        double small = StepSize(1e-4);
        double large = StepSize(1e-2);
        Assert.InRange(large / small, 50.0, 200.0);
    }
}
