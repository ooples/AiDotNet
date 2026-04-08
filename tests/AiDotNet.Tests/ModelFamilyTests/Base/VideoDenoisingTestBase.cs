using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for video denoising models. Inherits video NN invariants
/// and adds denoising-specific: clean input preserved and output bounded.
/// </summary>
public abstract class VideoDenoisingTestBase : VideoNNModelTestBase
{
    [Fact]
    public void CleanInput_ShouldBePreserved()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var cleanInput = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(cleanInput);

        // Denoising a clean signal should not add significant noise
        double mse = 0;
        int minLen = Math.Min(cleanInput.Length, output.Length);
        for (int i = 0; i < minLen; i++)
        {
            double diff = cleanInput[i] - output[i];
            mse += diff * diff;
        }
        mse /= Math.Max(1, minLen);

        // Loose threshold — just verify denoiser doesn't catastrophically corrupt clean input
        Assert.True(mse < 10.0,
            $"Denoising MSE = {mse:F4} on clean input — denoiser is corrupting clean signal.");
    }

    [Fact]
    public void DenoisedOutput_ShouldBeBounded()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Denoised output[{i}] is NaN — denoising introduced instability.");
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"Denoised output[{i}] = {output[i]:E4} is unbounded.");
        }
    }
}
