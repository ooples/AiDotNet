using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for audio classification models (genre, event, scene classification).
/// Inherits audio NN invariants and adds classification-specific: valid class outputs
/// and silence classification behavior.
/// </summary>
public abstract class AudioClassifierTestBase : AudioNNModelTestBase
{
    [Fact]
    public void ClassOutput_ShouldBeNonNegative()
    {
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

    [Fact]
    public void SilenceClassification_ShouldNotCrash()
    {
        var network = CreateNetwork();
        var silence = CreateConstantTensor(InputShape, 0.0);
        var output = network.Predict(silence);
        Assert.True(output.Length > 0, "Audio classifier produced empty output for silence.");
        for (int i = 0; i < output.Length; i++)
            Assert.False(double.IsNaN(output[i]), $"Audio class output[{i}] is NaN for silence.");
    }
}
