using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for video denoising models. Inherits video NN invariants
/// and adds denoising-specific: clean input preserved and output bounded.
/// </summary>
/// <remarks>
/// Generic over T so the source generator's float scaffold can emit
/// <c>VideoDenoisingTestBase&lt;float&gt;</c>. While this base was non-generic its models
/// (LiteDVDNet, ...) were locked to &lt;double&gt; and the float-first remedy was CS0308, leaving
/// only fixture shrinks/caps. Mirrors the FinancialModelTestBase/VideoNNModelTestBase pattern.
/// </remarks>
public abstract class VideoDenoisingTestBase<T> : VideoNNModelTestBase<T>
{
    [Fact(Timeout = 120000)]
    public async Task CleanInput_ShouldBePreserved()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        // A random pixel field is noise, not a clean video. Use an exactly black,
        // temporally constant sequence: it is a valid clean signal in every declared
        // pixel domain and gives this invariant a mathematically exact reference.
        // The strict MSE bound below is unchanged.
        var cleanInput = new Tensor<T>(InputShape);

        var output = network.Predict(cleanInput);

        // Denoising a clean signal should not add significant noise
        double mse = 0;
        int minLen = Math.Min(cleanInput.Length, output.Length);
        for (int i = 0; i < minLen; i++)
        {
            double diff = ConvertToDouble(cleanInput[i]) - ConvertToDouble(output[i]);
            mse += diff * diff;
        }
        mse /= Math.Max(1, minLen);

        // Loose threshold — just verify denoiser doesn't catastrophically corrupt clean input
        Assert.True(mse < 10.0,
            $"Denoising MSE = {mse:F4} on clean input — denoiser is corrupting clean signal.");
    }

    [Fact(Timeout = 120000)]
    public async Task DenoisedOutput_ShouldBeBounded()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            double v = ConvertToDouble(output[i]);
            Assert.False(double.IsNaN(v),
                $"Denoised output[{i}] is NaN — denoising introduced instability.");
            Assert.True(Math.Abs(v) < 1e6,
                $"Denoised output[{i}] = {v:E4} is unbounded.");
        }
    }
}

/// <summary>
/// Non-generic double-precision shim so existing <c>: VideoDenoisingTestBase</c> derivations
/// keep compiling (same pattern as VideoNNModelTestBase).
/// </summary>
public abstract class VideoDenoisingTestBase : VideoDenoisingTestBase<double> { }
