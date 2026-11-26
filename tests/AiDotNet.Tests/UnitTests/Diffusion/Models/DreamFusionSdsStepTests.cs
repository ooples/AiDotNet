using AiDotNet.Diffusion.ThreeD;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

public class DreamFusionSdsStepTests
{
    [Theory]
    [InlineData(4)]
    [InlineData(17)]
    public void SdsStep_IsOneSgdUpdateOfTheSummedImageObjective(int resolution)
    {
        // 17 x 17 crosses the 256-ray boundary with a short, differently weighted final chunk.
        var nerf = new NeRFNetwork<double>(hiddenDim: 4, numLayers: 1, seed: 7);
        var pose = new CameraPose { Azimuth = 45, Elevation = 15, Radius = 1.2 };
        var residual = new double[resolution * resolution * 3];
        for (int i = 0; i < residual.Length; i++)
            residual[i] = (i / 3 < 256 ? 0.05 : 0.75) + (0.1 * (i % 3));
        var gradient = new Tensor<double>(residual, new[] { resolution, resolution, 3 });
        var original = nerf.GetParameters().ToArray();
        var indices = new[] { 0, original.Length - 3, original.Length - 1 };
        var derivatives = new double[indices.Length];
        const double epsilon = 1e-5;
        const double learningRate = 1e-4;

        double Objective()
        {
            var image = nerf.Render(pose, resolution).ToArray();
            double value = 0;
            for (int i = 0; i < image.Length; i++) value += image[i] * residual[i];
            return value;
        }

        for (int i = 0; i < indices.Length; i++)
        {
            var perturbed = (double[])original.Clone();
            perturbed[indices[i]] += epsilon;
            nerf.SetParameters(new Vector<double>(perturbed));
            double plus = Objective();
            perturbed[indices[i]] -= 2 * epsilon;
            nerf.SetParameters(new Vector<double>(perturbed));
            derivatives[i] = (plus - Objective()) / (2 * epsilon);
        }
        Assert.True(derivatives.Any(value => Math.Abs(value) > 1e-6));

        nerf.SetParameters(new Vector<double>(original));
        nerf.UpdateParameters(gradient, pose, learningRate);
        var updated = nerf.GetParameters().ToArray();
        for (int i = 0; i < indices.Length; i++)
        {
            double expected = original[indices[i]] - (learningRate * derivatives[i]);
            Assert.InRange(Math.Abs(updated[indices[i]] - expected), 0, 2e-8);
        }
    }
}
