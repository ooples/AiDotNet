using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for audio classification models (genre, event, scene classification).
/// Inherits audio NN invariants and adds classification-specific: valid class outputs
/// and silence classification behavior.
/// </summary>
public abstract class AudioClassifierTestBase : AudioNNModelTestBase
{
    [Fact(Timeout = 60000)]
    public async Task ClassOutput_ShouldBeNonNegative()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(output[i] >= -1e-10,
                $"Audio class output[{i}] = {output[i]:F4} is negative — invalid class score.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task SilenceClassification_ShouldNotCrash()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        var silence = CreateConstantTensor(InputShape, 0.0);
        var output = network.Predict(silence);
        Assert.True(output.Length > 0, "Audio classifier produced empty output for silence.");
        for (int i = 0; i < output.Length; i++)
            Assert.False(double.IsNaN(output[i]), $"Audio class output[{i}] is NaN for silence.");
    }
}
